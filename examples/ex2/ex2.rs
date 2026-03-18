/*
    Examples 2
    
    Solve the incompressible lid-driven problem

*/

mod momentum;
mod pressure;

use std::collections::HashMap;

use finite_volumes::{core::mesh::PatchIndex, prelude::*};


enum ProblemType {
    LidDriven,
    Poiseuille,
}

#[derive(Clone, Copy, Debug)]
enum BoundaryCondition<const DIM: usize> {
    Inlet{velocity: Vector<DIM>},
    Wall,
    MovingWall{wall_velocity: Vector<DIM>},
    Outlet{pressure: f64},
}

struct BoundaryConditionSet<const DIM: usize> {
    bcs: HashMap<PatchIndex, BoundaryCondition<DIM>>,
}

impl<const DIM: usize> BoundaryConditionSet<DIM> {
    fn new() -> Self {
        Self { bcs: HashMap::new() }
    }
    pub fn with(mut self, bid: PatchIndex, bc: BoundaryCondition<DIM>) -> Self {
        self.bcs.insert(bid, bc);
        self
    }
    fn get(&self, bid: PatchIndex) -> Option<&BoundaryCondition<DIM>> {
        self.bcs.get(&bid)
    }
    fn velocity<'a>(&'a self)-> impl Fn(&FaceRef<'a, DIM>) -> (f64, Vector<DIM>) {
        |face| {
            let bid = face.boundary().unwrap();
            match self.get(bid).unwrap() {
                BoundaryCondition::Inlet { velocity } => {
                    (0.0, *velocity)
                },
                BoundaryCondition::Outlet { pressure: _ } => {
                    (1.0, Vector::zero())
                },
                BoundaryCondition::Wall => {
                    (0.0, Vector::zero())
                },
                BoundaryCondition::MovingWall { wall_velocity } => {
                    (0.0, *wall_velocity )
                }
            }
        }
    }
    fn pressure<'a>(&'a self) -> impl Fn(&FaceRef<'a, DIM>) -> (f64, f64) {
        |face| {
            let bid = face.boundary().unwrap();
            match self.get(bid).unwrap() {
                BoundaryCondition::Inlet { velocity: _ } => {
                    (1.0, 0.0)
                },
                BoundaryCondition::Outlet { pressure } => {
                    (0.0, *pressure)
                },
                BoundaryCondition::Wall => {
                    (1.0, 0.0)
                },
                BoundaryCondition::MovingWall { wall_velocity: _ } => {
                    (1.0, 0.0)
                }
            }
        }
    }
}



fn ex2<const DIM: usize>(
    problem: ProblemType,
    world: MpiCommunicator,
    viscosity: f64,
    dt: f64,
    steps: usize,
) -> Result<(), finite_volumes::error::Error> {

    // create the mesh
    let rank = world.rank() as usize;

    let mesh: Mesh<DIM> = Mesh::read(std::io::BufReader::new(std::fs::File::open(if world.size() == 1 {"examples/ex2/mesh.msh".to_string()} else {format!("examples/ex2/mesh_{}.msh", rank)}.as_str()).unwrap()), Some(world)).unwrap();

    // setup the boundary conditions
    let mut wall_velocity = Vector::zero();
    wall_velocity[0] = 1.0;
    let bcs = match problem {
        ProblemType::LidDriven => {
            BoundaryConditionSet::new()
            .with(mesh.patch_id("top").unwrap(), BoundaryCondition::MovingWall { wall_velocity })
            .with(mesh.patch_id("bottom").unwrap(), BoundaryCondition::Wall)
            .with(mesh.patch_id("left").unwrap(), BoundaryCondition::Wall)
            .with(mesh.patch_id("right").unwrap(), BoundaryCondition::Wall)
        },
        ProblemType::Poiseuille => {
            BoundaryConditionSet::new()
            .with(mesh.patch_id("top").unwrap(), BoundaryCondition::Wall)
            .with(mesh.patch_id("bottom").unwrap(), BoundaryCondition::Wall)
            .with(mesh.patch_id("left").unwrap(), BoundaryCondition::Inlet { velocity: wall_velocity })
            .with(mesh.patch_id("right").unwrap(), BoundaryCondition::Outlet { pressure: 0.0 })
        }
    };

    // create fields
    let mut velocity = Field::<Vector<DIM>, geometry::Cell, DIM>::from_mesh(&mesh);
    let mut velocity_gradient = Field::<Matrix<DIM, DIM>, geometry::Cell, DIM>::from_mesh(&mesh);
    let mut old_velocity = Field::<Vector<DIM>, geometry::Cell, DIM>::from_mesh(&mesh);

    let mut pressure = Field::<f64, geometry::Cell, DIM>::from_mesh(&mesh);
    let mut pressure_gradient = Field::<Vector<DIM>, geometry::Cell, DIM>::from_mesh(&mesh);

    let mut phi = Field::<f64, geometry::Face, DIM>::from_mesh(&mesh);

    let mut hbya = Field::<Vector<DIM>, geometry::Cell, DIM>::from_mesh(&mesh);
    let mut ainv = Field::<f64, geometry::Cell, DIM>::from_mesh(&mesh);
    let mut hbyan_face = Field::<f64, geometry::Face, DIM>::from_mesh(&mesh);
    let mut ainv_face = Field::<f64, geometry::Face, DIM>::from_mesh(&mesh);


    // init the velocity gradient
    momentum::compute_velocity_gradients(&mut velocity_gradient, &velocity, &mesh, bcs.velocity());

    // create a communicator
    let comm = Communicator::<geometry::Cell, _>::from_mesh(&mesh);

    // time iterations
    for time_iter in 1..=steps {

        if rank == 0 {println!("=== Time iter {} / {} ===", time_iter, steps)};
        old_velocity.set_from(velocity.raw_data());

        // assemble the momentum equation
        let (mlhs, mrhs) = momentum::assemble_momentum_equation(
            &mesh, 
            &velocity, 
            &velocity_gradient, 
            &phi, 
            viscosity, 
            dt,
            bcs.velocity(),
        )?;
        
        let npcorr = 3;
        for pcorr in 0..npcorr {

            // compute hbya and ainv
            pressure::compute_hbya_ainv(
                &mut hbya, 
                &mut ainv, 
                &velocity, 
                &mlhs, 
                &mrhs, 
                &mesh
            );

            // interpolate it on the faces
            pressure::intepolate_hbya_ainv_faces(
                &mut hbyan_face, 
                &mut ainv_face, 
                &hbya, 
                &ainv, 
                bcs.velocity(),
                bcs.pressure(),
                &mesh
            );

            // assemble the pressure equation
            let (plhs, prhs) = pressure::assemble_pressure_equation(
                &hbyan_face, 
                &ainv_face, 
                bcs.pressure(),
                &mesh
            )?;

            // solve the pressure equation
            {
                let mut solution = DistributedVector::from_data(pressure.raw_data());
                let precond = IncompleteCholesky::from_matrix(&plhs, 2);
                let result = solvers::conjugate_gradient(
                    &mut solution, 
                    &plhs, 
                    &prhs, 
                    &precond, 
                    &comm, 
                    1e-6, 
                    if pcorr == (npcorr - 1) {1e-3} else {0.1},
                    1000,
                ).unwrap();
                pressure.set_from(solution.data());
                if rank == 0 {println!("  solve pressure: {}", result);}
            }
            
            // compute pressure gradient
            pressure::compute_pressure_gradients(
                &mut pressure_gradient, 
                &pressure, 
                bcs.pressure(),
                &mesh
            );

            // correct the velocity
            pressure::correct_velocity(
                &mut velocity, 
                &hbya, 
                &ainv, 
                &pressure_gradient, 
                &mesh,
            );
        }

        // finished, correct face flux and velocity gradients
        pressure::correct_phi(
            &mut phi, 
            &hbyan_face, 
            &ainv_face, 
            &pressure, 
            bcs.pressure(),
            &mesh
        );

        momentum::compute_velocity_gradients(
            &mut velocity_gradient, 
            &velocity, 
            &mesh, 
            bcs.velocity()
        );
        

        // compute the residual
        let mut residual = 0.0;
        for cell in mesh.iter_cells() {
            residual += (velocity[cell.id()] - old_velocity[cell.id()]).norm().powi(2);
        }
        residual = (comm.single().reduce_add(residual) / comm.single().reduce_add(mesh.n_cells() as f64)).sqrt(); // / dt;
        if rank == 0 {println!("  residual = {:.3e}", residual)}

        if rank == 0 {println!();}

        if residual < 1e-5 {break}

    }

    // done! write solution
    PvtuWriter::new(&mesh)
        .with("p", &pressure)
        .with("U", &velocity)
        .write("examples/ex2/solution.pvtu")
        .unwrap();

    Ok(())
}




fn main() -> Result<(), finite_volumes::error::Error> {

    let universe = mpi::initialize().ok_or(finite_volumes::error::Error::MpiInitializeFailed)?;
    let world = universe.world();

    let args: Vec<String> = std::env::args().collect();
    let problem = if args.len() == 1 {
        ProblemType::LidDriven
    } else {
        match args[1].as_str() {
            "lid-driven" => ProblemType::LidDriven,
            "poiseuille" => ProblemType::Poiseuille,
            _ => panic!("Problem type {} invalid", args[1])
        }
    };

    ex2::<2>(
        problem,
        world,
        1.0 / 2000.0,
        0.025,
        1000,
    )?;

    Ok(())
}



