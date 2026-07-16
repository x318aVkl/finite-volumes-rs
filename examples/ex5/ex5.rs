/*
    LES of flow past a backface
*/

use finite_volumes::{fvm::{schemes::faceinterp::FaceInterpolationScheme, tools::{gradients::compute_gradients, hbya::{compute_hbya_ainv, correct_phi, correct_velocity, intepolate_hbya_ainv_faces}, limiters::compute_limiters}}, prelude::*};

struct Parameters {
    laminar_viscoisty: f64,
    velocity: f64,
    time_step: f64,
    cfl: f64,
    write_interval: f64,
    time_iterations: usize,
    momentum_predictor: bool,
    pressure_correctors: usize,
    pressure_linear_options: LinearSolverOptions,
    pressure_linear_options_final: LinearSolverOptions,
    smagorinsky_cs: f64,
}


fn noise<const DIM: usize>(x: Vector<DIM>, time: f64) -> f64 {
    let mut rnd = 0.0;
    let seeds = [2981.3, 90281.9, 1928.9, 817.9];
    let factors = [0.7, 1.4, 0.78, 1.1];
    let factors_time = [0.7, 1.4, 0.78, 1.1];
    let ftime = 200.0;
    for dim in 0..DIM {
        let mut f = 10.0;
        let mut a = 0.2;
        for k in 0..factors.len() {
            rnd += ((x[dim] * factors[k] + seeds[k] + factors_time[k] * time * ftime) * f).sin() * a;
            f *= 2.0;
            a *= 0.6;
        }
    }
    rnd
}
fn noise_vec<const DIM: usize>(x: Vector<DIM>, time: f64) -> Vector<DIM> {
    let mut out = Vector::new();
    out[0] = noise(x, time);
    out[1] = noise(x + Vector::one()*2.23, time + 17.3);
    out[2] = noise(x + Vector::one()*5.1, time + 281.3);
    out
}

fn ex5<const DIM: usize>(
    parameters: Parameters,
    world: MpiCommunicator,
) -> Result<(), Box<dyn std::error::Error>> {

    let rank = world.rank();

    let mesh: Mesh<DIM> = Mesh::read(std::fs::File::open(if world.size() == 1 {"examples/ex5/mesh.msh".to_string()} else {format!("examples/ex5/mesh_{}.msh", rank)}.as_str()).unwrap(), Some(world)).unwrap();

    // create the fields
    let mut velocity = Field::<Vector<DIM>, geometry::Cell, DIM>::from_mesh(&mesh);
    let mut pressure = Field::<f64, geometry::Cell, DIM>::from_mesh(&mesh);
    let mut turbulent_viscosity = Field::<f64, geometry::Cell, DIM>::from_mesh(&mesh);

    let mut viscosity = Field::<f64, geometry::Face, DIM>::from_mesh(&mesh);
    let mut phi = Field::<f64, geometry::Face, DIM>::from_mesh(&mesh);

    // initialize cell fields
    for cell in mesh.iter_cells() {
        let mut vi = Vector::<DIM>::zero();
        vi[0] = parameters.velocity;
        velocity[cell.id()] = vi;
    }
    velocity.update();

    let walls_bot = mesh.patch_id("walls_bot").unwrap();
    let walls_top = mesh.patch_id("walls_top").unwrap();
    let inlet = mesh.patch_id("inlet").unwrap();
    let outlet = mesh.patch_id("outlet").unwrap();
    let sides = mesh.patch_id("sides").unwrap();

    // initialize the face fields
    for face in mesh.iter_faces() {
        viscosity[face.id()] = parameters.laminar_viscoisty;
        let is_walls_bot = match face.boundary() {
            Some(b) => {
                b == walls_bot
            },
            None => false,
        };
        if is_walls_bot {
            phi[face.id()] = 0.0;
        } else {
            let vi = Vector::<DIM>::zero();
            //vi[0] = parameters.velocity;
            phi[face.id()] = vi.dot(face.normal());
        }
    }
    viscosity.update();
    phi.update();

    // create the gradients fields
    let mut velocity_grad = Field::<Matrix<DIM, DIM>, geometry::Cell, DIM>::from_mesh(&mesh);
    let mut velocity_lim = Field::<f64, geometry::Face, DIM>::from_mesh(&mesh);

    let mut pressure_grad = Field::<Vector<DIM>, geometry::Cell, DIM>::from_mesh(&mesh);

    // initialize the boundary conditions
    let mut velocity_constraints = FaceConstraints::<Vector<DIM>, f64>::from_mesh(&mesh);
    let mut pressure_constraints = FaceConstraints::<f64, f64>::from_mesh(&mesh);
    
    for face in mesh.patch(inlet).iter() {
        let mut vi = Vector::zero();
        vi[0] = parameters.velocity;
        velocity_constraints[face.id()] = (0.0, vi);
        pressure_constraints[face.id()] = (1.0, 0.0);
    }
    for face in mesh.patch(outlet).iter() {
        velocity_constraints[face.id()] = (1.0, Vector::zero());
        pressure_constraints[face.id()] = (0.0, 0.0);
    }
    for face in mesh.patch(walls_top).iter() {
        velocity_constraints[face.id()] = (1.0, Vector::zero());
        pressure_constraints[face.id()] = (1.0, 0.0);
    }
    for face in mesh.patch(walls_bot).iter() {
        velocity_constraints[face.id()] = (0.0, Vector::zero());
        pressure_constraints[face.id()] = (1.0, 0.0);
    }
    for face in mesh.patch(sides).iter() {
        velocity_constraints[face.id()] = (1.0, Vector::zero());
        pressure_constraints[face.id()] = (1.0, 0.0);
    }

    let mut time = 0.0;
    let mut dt = parameters.time_step;
    let mut dt_last: f64;
    let cfl = parameters.cfl;

    // hbya and ainv fields
    let mut hbya = Field::<Vector<DIM>, geometry::Cell, DIM>::from_mesh(&mesh);
    let mut ainv = Field::<f64, geometry::Cell, DIM>::from_mesh(&mesh);

    let mut hbyan_face = Field::<f64, geometry::Face, DIM>::from_mesh(&mesh);
    let mut ainv_face = Field::<f64, geometry::Face, DIM>::from_mesh(&mesh);

    let comm = Communicator::<geometry::Cell, DIM>::from_mesh(&mesh);

    let mut write_iter = 0;
    let mut next_write_time = parameters.write_interval;

    let mut velocity_last = velocity.clone();

    // time loop thingy
    for time_iter in 1..=parameters.time_iterations {
        if rank == 0 {println!("=== iter: {}, time: {:.6} ===", time_iter, time);}

        dt_last = dt;

        // adjust the time step
        let mut time_to_write = false;
        let mut min_dt: f64 = 1e20;
        for cell in mesh.iter_cells() {
            let ucell = velocity[cell.id()].norm();
            let dx = cell.volume().powf(1.0 / (DIM as f64));
            let dti = cfl * dx / ucell.max(1e-15);
            min_dt = min_dt.min(dti);
        }
        let min_dt = comm.single().reduce_min(min_dt);
        let new_dt = if min_dt > dt {min_dt.min(dt * 1.2)} else {min_dt};
        let dt_pre_adjust_write = new_dt;
        if (time + new_dt) > next_write_time {
            dt = next_write_time - time;
            time_to_write = true;
        } else {
            dt = new_dt;
        }
        if rank == 0 {println!("- dt: {:.6}", dt);}
    
        let (mlhs, mrhs) = assembly::assemble::<Vector<DIM>, f64, Vector<DIM>, DIM>(
            terms::time(schemes::time::Backward::new(&velocity, &velocity_last, dt, dt_last))
                + terms::convection(schemes::faceinterp::LimitedLinear::new(&phi, &velocity_lim), &phi)
                - terms::laplacian(schemes::facengrad::Orthogonal::new(), &viscosity)
            ,
            velocity_constraints.as_bc(),
            &mesh,
        );
        velocity_last = velocity.clone();

        if parameters.momentum_predictor {
            // solve the momentum predictor
        }

        for pcorr in 1..=parameters.pressure_correctors {
            compute_hbya_ainv(&mut hbya, &mut ainv, &velocity, &mlhs, &mrhs, &mesh);
            intepolate_hbya_ainv_faces::<f64, DIM>(
                &mut hbyan_face, &mut ainv_face, &hbya, &ainv, 
                velocity_constraints.as_bc(), pressure_constraints.as_bc(), 
                schemes::faceinterp::Linear::new(), 
                schemes::faceinterp::Linear::new(), 
                &mesh
            );

            let (lhs, rhs) = assembly::assemble(
                    - terms::laplacian(schemes::facengrad::Orthogonal::new(), &ainv_face)
                    + terms::divergence::<f64, f64, f64, _>(&hbyan_face)
                ,
                pressure_constraints.as_bc(),
                &mesh,
            );

            let mut solution = DistributedVector::from_data(pressure.raw_data());
            let precond = preconditioners::IncompleteCholesky::from_matrix(&lhs, 0);
            let result = solvers::conjugate_gradient(
                &mut solution, 
                &lhs, &rhs, &precond, 
                &comm, 
                if pcorr == parameters.pressure_correctors {
                    parameters.pressure_linear_options_final
                } else {
                    parameters.pressure_linear_options
                }
            )?;
            if rank == 0 { println!("- solved for pressure: {}", result); }
            pressure.set_from(solution.data());
            compute_gradients(
                &mut pressure_grad, 
                &pressure,
                pressure_constraints.as_bc(),
                &mesh, 
            );

            // correct hbya
            correct_velocity(
                &mut velocity, 
                &hbya, &ainv, 
                &pressure_grad, 
                &mesh
            );
            // add sinusoidal noise to the velocity inlet
            for face in mesh.patch(inlet).iter() {
                let x = face.center();
                let mut rnd = noise_vec(x, time);
                rnd[0] += parameters.velocity;
                velocity_constraints[face.id()] = (0.0, rnd);
            }
            // also correct the velocity face constraints for slip boundary condition
            for face in mesh.patch(walls_top).iter() {
                let n = face.normal();
                let ucell = velocity[face.owner()];
                velocity_constraints[face.id()] = (1.0, Vector::zero() - n * ucell.dot(n));
            }
            for face in mesh.patch(sides).iter() {
                let n = face.normal();
                let ucell = velocity[face.owner()];
                velocity_constraints[face.id()] = (1.0, Vector::zero() - n * ucell.dot(n));
            }

        }

        // at the end of the pressure correction loop, update velocity gradients and limiters, and phi
        correct_phi(
            &mut phi, 
            &hbyan_face, &ainv_face, &pressure, 
            schemes::facengrad::Orthogonal::new(), 
            pressure_constraints.as_bc(), 
            &mesh
        );
        compute_gradients(
            &mut velocity_grad,
            &velocity,
            velocity_constraints.as_bc(),
            &mesh,
        );
        compute_limiters::<Vector<DIM>, Matrix<DIM, DIM>, f64, Vector<DIM>, f64, f64, DIM>(
            &mut velocity_lim,
            &velocity,
            &velocity_grad,
            schemes::limiters::LimitedLinear(1.0),
            schemes::faceinterp::Linear::new(),
            schemes::facengrad::Orthogonal::new(),
            velocity_constraints.as_bc(),
            &mesh,
        );

        // done! now update the turbulent viscosity and overall viscosity
        for cell in mesh.iter_cells() {
            let u_grad = velocity_grad[cell.id()];
            let s_ij = 0.5 * (u_grad + u_grad.transpose());
            let s_norm = (2.0 * s_ij.sumsq()).sqrt();
            let delta = cell.volume().powf(1.0 / (DIM as f64));
            let mu_t = (parameters.smagorinsky_cs * delta).powi(2) * s_norm;
            turbulent_viscosity[cell.id()] = mu_t;
        }
        turbulent_viscosity.update();
        let mu_interp = schemes::faceinterp::Linear::<f64, f64, DIM>::new();
        for face in mesh.iter_faces() {
            let mut0 = turbulent_viscosity[face.owner()];
            let mutface = match face.neighbor() {
                FaceNeighbor::Boundary(b) => {if b == walls_bot {0.0} else {mut0}},
                FaceNeighbor::Cell(c1) => {
                    let mut1 = turbulent_viscosity[c1];
                    let (t0, t1, tr) = mu_interp.terms(&face, &mesh);
                    t0 * mut0 + t1 * mut1 + tr
                }
            };
            viscosity[face.id()] = mutface + parameters.laminar_viscoisty;
        }
        viscosity.update();

        time += dt;

        // now if time to write, write
        if time_to_write {
            PvtuWriter::new(&mesh)
                .with("U", &velocity)
                .with("p", &pressure)
                .with("mu_t", &turbulent_viscosity)
                .write(format!("examples/ex5/data/solution_{}.pvtu", write_iter).as_str())?;
            write_iter += 1;
            next_write_time += parameters.write_interval;
            dt = dt_pre_adjust_write;
        }

    }

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let universe = mpi::initialize().ok_or(finite_volumes::error::Error::MpiInitializeFailed)?;
    let world = universe.world();

    let parameters = Parameters {
        laminar_viscoisty: 0.0001,
        velocity: 3.0,
        time_step: 1e-4,
        cfl: 0.8,
        write_interval: 0.05,
        time_iterations: 10000,
        momentum_predictor: false,
        pressure_correctors: 5,
        smagorinsky_cs: 0.18,
        pressure_linear_options: LinearSolverOptions { 
            relative_tolerance: 0.1, 
            absolute_tolerance: 1e-5, 
            max_iterations: 500,
        },
        pressure_linear_options_final: LinearSolverOptions { 
            relative_tolerance: 1e-3, 
            absolute_tolerance: 1e-5, 
            max_iterations: 500,
        },
    };

    ex5::<3>(parameters, world)
}