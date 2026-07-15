/*
    Example 1

    Solve the convection-diffusion equation

*/

use finite_volumes::prelude::*;


fn ex1<const DIM: usize>(world: MpiCommunicator) -> Result<(), finite_volumes::error::Error> {

    // create the mesh
    let rank = world.rank() as usize;

    let mesh: Mesh<2> = Mesh::read(std::fs::File::open(if world.size() == 1 {"examples/ex1/mesh.msh".to_string()} else {format!("examples/ex1/mesh_{}.msh", rank)}.as_str()).unwrap(), Some(world)).unwrap();

    // compute and store the face flux for advection
    let mut flux = Field::<f64, geometry::Face, _>::from_mesh(&mesh);
    let mut diffusivity = Field::<f64, geometry::Face, _>::from_mesh(&mesh);
    for face in mesh.iter_faces() {
        let velocity: Vector<_> = [1.0, 1.0].into();
        flux[face.id()] = face.normal().dot(velocity);
        diffusivity[face.id()] = 0.005;
    }
    flux.update();
    diffusivity.update();

    // solution field
    let mut field = Field::<f64, geometry::Cell, _>::from_mesh(&mesh);
    let mut gradients = Field::<Vector<2>, geometry::Cell, _>::from_mesh(&mesh);
    let mut limiters = Field::<f64, geometry::Face, _>::from_mesh(&mesh);

    let bc = |face: &FaceRef<2>| {
        let flux = flux[face.id()];
        if flux < 0.0 {
            (0.0, 1.0)
        } else {
            (1.0, 0.0)
        }
    };

    let dt = 0.1;

    for time_iter in 1..=100 {

        // assemble poisson equation with source term and solve
        {
            let (lhs, rhs) = assembly::assemble(
                terms::time(schemes::time::Euler::new(&field, dt))
                    + terms::convection(schemes::faceinterp::LimitedLinear::new(&flux, &limiters), &flux)
                    - terms::laplacian(schemes::facengrad::Corrected::new(&gradients, 1.0), &diffusivity)
                ,
                bc,
                &mesh
            );

            let comm = Communicator::<geometry::Cell, _>::from_mesh(&mesh);

            // solve
            let mut solution = DistributedVector::from_data(field.raw_data());

            let precond = IncompleteLowerUpper::from_matrix(&lhs, 1);
            let result = solvers::bi_conjugate_gradient_stab(
                &mut solution,
                &lhs,
                &rhs,
                &precond,
                &comm,
                LinearSolverOptions::default(),
            ).unwrap();

            if rank == 0 {println!("iter {}, solved: {}", time_iter, result);}
            
            field.set_from(solution.data());

            if result.initial_residual < 1e-4 {
                break;
            }
        }

        // compute and update gradients
        tools::gradients::compute_gradients(
            &mut gradients, 
            &mut field, 
            bc, 
            &mesh
        );

        tools::limiters::compute_limiters::<f64, Vector<2>, f64, f64, f64, f64, _>(
            &mut limiters, 
            &field, 
            &gradients, 
            schemes::limiters::LimitedLinear(1.0), 
            schemes::faceinterp::Linear::new(), 
            schemes::facengrad::Corrected::new(&gradients, 1.0), 
            bc, 
            &mesh
        );

    }

    // write the solution
    PvtuWriter::new(&mesh)
        .with("phi", &field)
        .write("examples/ex1/solution.pvtu")
        .unwrap();


    Ok(())
}


fn main() -> Result<(), finite_volumes::error::Error> {

    let universe = mpi::initialize().ok_or(finite_volumes::error::Error::MpiInitializeFailed)?;
    let world = universe.world();

    ex1::<2>(world)?;

    Ok(())
}


