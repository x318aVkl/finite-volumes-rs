/*
    Examples 2
    
    Solve the incompressible lid-driven problem

*/

mod momentum;
mod pressure;


use finite_volumes::prelude::*;




fn ex2<const DIM: usize>(
    world: MpiCommunicator,
    viscosity: f64,
    dt: f64,
    steps: usize,
) -> Result<(), finite_volumes::error::Error> {

    // create the mesh
    let rank = world.rank() as usize;

    let mesh: Mesh<DIM> = Mesh::read(std::io::BufReader::new(std::fs::File::open(if world.size() == 1 {"examples/ex2/mesh.msh".to_string()} else {format!("examples/ex2/mesh_{}.msh", rank)}.as_str()).unwrap()), Some(world)).unwrap();

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
    momentum::compute_velocity_gradients(&mut velocity_gradient, &velocity, &mesh);

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
            dt
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
                &mesh
            );

            // assemble the pressure equation
            let (plhs, prhs) = pressure::assemble_pressure_equation(
                &hbyan_face, 
                &ainv_face, 
                &mesh
            )?;

            // solve the pressure equation
            {
                let mut solution = DistributedVector::from_data(pressure.raw_data());
                let precond = IncompleteCholesky::from_matrix(&plhs);
                let result = solvers::conjugate_gradient(
                    &mut solution, 
                    &plhs, 
                    &prhs, 
                    &precond, 
                    &comm, 
                    if pcorr == (npcorr - 1) {1e-8} else {1e-4}, 
                    1000,
                ).unwrap();
                pressure.set_from(solution.data());
                if rank == 0 {println!("  solve pressure: {}", result);}
            }
            
            // compute pressure gradient
            pressure::compute_pressure_gradients(
                &mut pressure_gradient, 
                &pressure, 
                &mesh
            );

            // correct the face flux
            pressure::correct_phi(
                &mut phi, 
                &hbyan_face, 
                &ainv_face, 
                &pressure, 
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

        // compute the residual
        let mut residual = 0.0;
        for cell in mesh.iter_cells() {
            residual += (velocity[cell.id()] - old_velocity[cell.id()]).norm().powi(2);
        }
        residual = (comm.single().reduce_add(residual) / comm.single().reduce_add(mesh.n_cells() as f64)).sqrt(); // / dt;
        if rank == 0 {println!("  residual = {:.3e}", residual)}

        if residual < 1e-5 {break}

    }


    // done! write solution
    PvtuWriter::new(&mesh)
        .with_scalar("p", &pressure)
        .with_vector("U", &velocity)
        .write("examples/ex2/solution.pvtu")
        .unwrap();

    Ok(())
}




fn main() -> Result<(), finite_volumes::error::Error> {

    let universe = mpi::initialize().ok_or(finite_volumes::error::Error::MpiInitializeFailed)?;
    let world = universe.world();

    ex2::<2>(
        world,
        1.0 / 100.0,
        0.1,
        100,
    )?;

    Ok(())
}



