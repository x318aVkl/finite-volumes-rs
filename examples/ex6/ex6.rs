/*
    Two-phase flow using VOF
*/

use mpi::topology::SimpleCommunicator;

use finite_volumes::{fvm::{schemes::faceinterp::FaceInterpolationScheme, tools::{gradients::compute_gradients, hbya::{compute_hbya_ainv, correct_phi, correct_velocity, intepolate_hbya_ainv_faces}, limiters::compute_limiters}}, prelude::*};


struct Parameters {
    velocity: f64,
    viscosity: f64,
    fluid_density: f64,
    gas_density: f64,
    surface_tension: f64,
    time_step: f64,
    cfl: f64,
    time_iterations: usize,
    outer_correctors: usize,
    momentum_predictor: bool,
    pressure_correctors: usize,
    pressure_linear_options: LinearSolverOptions,
    pressure_linear_options_final: LinearSolverOptions,
    alpha_correctors: usize,
    alpha_linear_options: LinearSolverOptions,
}


fn ex6<const DIM: usize>(
    world: SimpleCommunicator,
    parameters: Parameters,
) -> Result<(), Box<dyn std::error::Error>> {
    let rank = world.rank();

    let mesh: Mesh<DIM> = Mesh::read(std::fs::File::open(if world.size() == 1 {"examples/ex6/mesh.msh".to_string()} else {format!("examples/ex6/mesh_{}.msh", rank)}.as_str()).unwrap(), Some(world)).unwrap();

    let hdim = if DIM == 2 {1} else {2};

    // create the fields
    let mut velocity = Field::<Vector<DIM>, geometry::Cell, DIM>::from(&mesh);
    let mut pressure = Field::<f64, geometry::Cell, DIM>::from(&mesh);

    let mut velocity_grad = Field::<Matrix<DIM, DIM>, geometry::Cell, DIM>::from(&mesh);
    let mut velocity_lim = Field::<f64, geometry::Face, DIM>::from(&mesh);

    let mut pressure_grad = Field::<Vector<DIM>, geometry::Cell, DIM>::from(&mesh);

    let mut alpha = Field::<f64, geometry::Cell, DIM>::from(&mesh);
    let mut alpha_grad = Field::<Vector<DIM>, geometry::Cell, DIM>::from(&mesh);
    let mut alpha_lim = Field::<f64, geometry::Face, DIM>::from(&mesh);
    let mut alpha_lim_comp = Field::<f64, geometry::Face, DIM>::from(&mesh);

    let mut viscosity = Field::<f64, geometry::Face, DIM>::from(&mesh);
    let mut phi = Field::<f64, geometry::Face, DIM>::from(&mesh);

    let mut density = Field::<f64, geometry::Cell, DIM>::from(&mesh);
    let mut density_face = Field::<f64, geometry::Face, DIM>::from(&mesh);
    let mut density_grad = Field::<Vector<DIM>, geometry::Cell, DIM>::from(&mesh);

    // initialize cell fields
    for cell in mesh.iter_cells() {
        let mut vi = Vector::<DIM>::zero();
        vi[0] = parameters.velocity;
        if cell.center()[hdim] < 0.0 {
            alpha[cell.id()] = 1.0;
            //vi[0] = parameters.velocity;
        } else {
            alpha[cell.id()] = 0.0;
        }
        velocity[cell.id()] = vi;
    }
    velocity.update();
    alpha.update();

    // extract the boundary labels
    let inlet = mesh.patch_id("inlet").unwrap();
    let outlet = mesh.patch_id("outlet").unwrap();
    let sides = mesh.patch_id("sides").unwrap();
    let wall_cylinder = mesh.patch_id("wall").unwrap();

    for face in mesh.iter_faces() {
        viscosity[face.id()] = parameters.viscosity;
        let mut vi = Vector::<DIM>::zero();
        vi[0] = parameters.velocity;
        let nslip = match face.boundary() {
            None => {
                false
            },
            Some(b) => b == wall_cylinder,
        };
        if nslip {
            phi[face.id()] = 0.0;
        } else {
            phi[face.id()] = vi.dot(face.normal());
        }
    }
    viscosity.update();
    phi.update();

    let mut gravity = Vector::<DIM>::zero();
    gravity[hdim] = -1.0;


    // boundary conditions
    let mut velocity_constraints = FaceConstraints::<Vector<DIM>, f64>::from(&mesh);
    let mut pressure_constraints = FaceConstraints::<f64, f64>::from(&mesh);
    let mut alpha_constraints = FaceConstraints::<f64, f64>::from(&mesh);
    for face in mesh.patch(inlet).iter() {
        let mut vi = Vector::zero();
        vi[0] = parameters.velocity;
        velocity_constraints[face.id()] = (0.0, vi);
        pressure_constraints[face.id()] = (1.0, 0.0);
        alpha_constraints[face.id()] = (0.0, if face.center()[hdim] < 0.0 {1.0} else {0.0});
    }
    for face in mesh.patch(outlet).iter() {
        velocity_constraints[face.id()] = (1.0, Vector::zero());
        pressure_constraints[face.id()] = (0.0, 0.0);
        alpha_constraints[face.id()] = (1.0, 0.0);
    }
    for face in mesh.patch(wall_cylinder).iter() {
        velocity_constraints[face.id()] = (0.0, Vector::zero());
        pressure_constraints[face.id()] = (1.0, 0.0);
        alpha_constraints[face.id()] = (1.0, 0.0);
    }
    for face in mesh.patch(sides).iter() {
        velocity_constraints[face.id()] = (0.0, Vector::zero());
        pressure_constraints[face.id()] = (1.0, 0.0);
        alpha_constraints[face.id()] = (1.0, 0.0);
    }

    // update the density field
    for cell in mesh.iter_cells() {
        let a = alpha[cell.id()];
        let rho = parameters.gas_density * (1.0 - a) + parameters.fluid_density * a;
        density[cell.id()] = rho;
    }
    density.update();
    // interpolate rho on faces
    {
        let interp = schemes::faceinterp::LimitedLinear::<f64, f64, DIM>::new(&phi, &alpha_lim);
        for face in mesh.iter_faces() {
            let (t0, t1, r) = interp.terms(&face, &mesh);
            match face.neighbor() {
                FaceNeighbor::Boundary(b) => {
                    density_face[face.id()] = density[face.owner()];
                    if b == inlet {
                        density_face[face.id()] = if face.center()[hdim] < 0.0 {parameters.fluid_density} else {parameters.gas_density};
                    }
                },
                FaceNeighbor::Cell(c1) => {
                    density_face[face.id()] = 
                        t0 * density[face.owner()]
                        + t1 * density[c1]
                        + r;
                }
            }
        }
    }
    density_face.update();
    for face in mesh.iter_faces() {
        phi[face.id()] *= density_face[face.id()];
    }
    phi.update();

    // loop thingy
    let mut time = 0.0;
    let mut dt = parameters.time_step;
    let mut dt_last: f64;
    let cfl = parameters.cfl;

    // hbya and ainv fields
    let mut hbya = Field::<Vector<DIM>, geometry::Cell, DIM>::from(&mesh);
    let mut ainv = Field::<f64, geometry::Cell, DIM>::from(&mesh);

    let mut hbyan_face = Field::<f64, geometry::Face, DIM>::from(&mesh);
    let mut ainv_face = Field::<f64, geometry::Face, DIM>::from(&mesh);

    let mut rho_hbyan_face = Field::<f64, geometry::Face, DIM>::from(&mesh);
    let mut rho_ainv_face = Field::<f64, geometry::Face, DIM>::from(&mesh);

    let comm = Communicator::<geometry::Cell, DIM>::from(&mesh);

    let mut velocity_last = velocity.clone();
    let mut velocity_last2 = velocity.clone();
    let mut alpha_last = alpha.clone();
    let mut alpha_last2 = alpha.clone();
    let mut density_last = density.clone();
    let mut density_last2 = density.clone();

    let mut surface_tension_source = Field::<Vector<DIM>, geometry::Cell, DIM>::from(&mesh);

    // field for density_grad .dot (h)
    let mut rhorgh = density_grad.clone();

    let mut write_iter = 0;

    for time_iter in 1..=parameters.time_iterations {
        if rank == 0 {println!("=== iter: {}, time: {:.6} ===", time_iter, time);}

        dt_last = dt;

        // adjust the time step
        let mut min_dt: f64 = 1e20;
        for face in mesh.iter_faces() {
            let dx = match face.neighbor() {
                FaceNeighbor::Boundary(_) => {
                    (face.center() - mesh.cell(face.owner()).center()).dot(face.normal()).abs()
                },
                FaceNeighbor::Cell(c1) => {
                    (mesh.cell(c1).center() - mesh.cell(face.owner()).center()).dot(face.normal()).abs()
                }
            };
            let uface = phi[face.id()].abs() / density_face[face.id()];
            let dti = cfl * dx / uface.max(1e-15);
            min_dt = min_dt.min(dti);
        }
        let min_dt = comm.single().reduce_min(min_dt);
        let new_dt = if min_dt > dt {min_dt.min(dt * 1.2)} else {min_dt};
        dt = new_dt;
        if rank == 0 {println!("- dt: {:.6}", dt);}

        velocity_last2 = velocity_last.clone();
        velocity_last = velocity.clone();

        density_last2 = density_last.clone();
        density_last = density.clone();

        alpha_last2 = alpha_last.clone();
        alpha_last = alpha.clone();


        for _outer_corr in 1..=parameters.outer_correctors {

            // update rhorgh
            for cell in mesh.iter_cells() {
                let mut h = Vector::zero();
                h[hdim] = cell.center()[hdim];
                let g_dot_h = gravity.dot(h);
                //rhorgh[cell.id()] = g_dot_h * density_grad[cell.id()];
                rhorgh[cell.id()] = gravity * density[cell.id()];
            }
        
            let (mlhs, mrhs) = assembly::assemble::<Vector<DIM>, f64, Vector<DIM>, DIM>(
                terms::time(schemes::time::Euler::new_with_density(&velocity_last, dt, &density, &density_last))
                    //+ terms::convection(schemes::faceinterp::LimitedLinear::new(&phi, &velocity_lim), &phi)
                    + terms::convection(schemes::faceinterp::LimitedLinear::new(&phi, &velocity_lim), &phi)
                    //- terms::laplacian(schemes::facengrad::Corrected::new(&velocity_grad, 1.0), &viscosity)
                    - terms::laplacian(schemes::facengrad::Corrected::new(&velocity_grad, 1.0), &viscosity)
                    - terms::source(&rhorgh)
                    - terms::source(&surface_tension_source)
                ,
                velocity_constraints.as_bc(),
                &mesh,
            );

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
                for face in mesh.iter_faces() {
                    rho_ainv_face[face.id()] = density_face[face.id()] * ainv_face[face.id()];
                    rho_hbyan_face[face.id()] = density_face[face.id()] * hbyan_face[face.id()];
                }

                let (lhs, rhs) = assembly::assemble(
                        - terms::laplacian(schemes::facengrad::Corrected::new(&pressure_grad, 1.0), &rho_ainv_face)
                        + terms::divergence::<f64, f64, f64, _>(&rho_hbyan_face)
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
                // also correct the velocity face constraints for slip boundary condition
                // for face in mesh.patch(sides).iter() {
                //     let n = face.normal();
                //     let ucell = velocity[face.owner()];
                //     velocity_constraints[face.id()] = (1.0, Vector::zero() - n * ucell.dot(n));
                // }

            }

            // at the end of the pressure correction loop, update velocity gradients and limiters, and phi
            correct_phi(
                &mut phi, 
                &rho_hbyan_face, &rho_ainv_face, &pressure, 
                schemes::facengrad::Corrected::new(&pressure_grad, 1.0), 
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
                schemes::faceinterp::Upwind::new(&phi),
                schemes::facengrad::Corrected::new(&velocity_grad, 1.0),
                velocity_constraints.as_bc(),
                &mesh,
            );


            for alpha_corr in 1..=parameters.alpha_correctors {

                let mut ucomp = phi.clone();
                let iascheme = schemes::faceinterp::LimitedLinear::<f64, f64, DIM>::new(&phi, &alpha_lim);
                let ischeme = schemes::faceinterp::Linear::<f64, f64, DIM>::new();
                for face in mesh.iter_faces() {
                    let u = phi[face.id()];
                    let (t0a, t1a, _) = iascheme.terms(&face, &mesh);
                    let (t0, t1, _) = ischeme.terms(&face, &mesh);
                    let a = match face.neighbor() {FaceNeighbor::Boundary(_) => alpha[face.owner()], FaceNeighbor::Cell(c1) => t0a * alpha[face.owner()] + t1a * alpha[c1]};
                    let a = a.max(0.0).min(1.0);
                    let a_grad = match face.neighbor() {FaceNeighbor::Boundary(_) => alpha_grad[face.owner()], FaceNeighbor::Cell(c1) => t0 * alpha_grad[face.owner()] + t1 * alpha_grad[c1]};
                    let t = a_grad / a_grad.norm().max(1e-14);
                    let mag = u.abs();
                    ucomp[face.id()] = (1.0 - a) * t.dot(face.normal()) * mag * 0.5 + u;
                }
                ucomp.update();

                compute_limiters::<f64, Vector<DIM>, f64, f64, f64, f64, DIM>(
                    &mut alpha_lim_comp,
                    &alpha,
                    &alpha_grad,
                    schemes::limiters::LimitedLinear(1.0),
                    schemes::faceinterp::Upwind::new(&ucomp),
                    schemes::facengrad::Corrected::new(&alpha_grad, 1.0),
                    alpha_constraints.as_bc(),
                    &mesh,
                );

                // done! now solve for phase fraction
                {
                    let (lhs, rhs) = assembly::assemble(
                            terms::time(schemes::time::Euler::new_with_density(&alpha_last, dt, &density, &density_last))
                            //+ terms::convection(schemes::faceinterp::LimitedLinear::new(&phi, &alpha_lim), &rphi)
                            + terms::convection(schemes::faceinterp::LimitedLinear::new(&ucomp, &alpha_lim_comp), &ucomp)
                            //+ terms::convection(schemes::faceinterp::Upwind::new(&ucomp), &ucomp)
                        ,
                        alpha_constraints.as_bc(),
                        &mesh,
                    );
                    let mut solution = DistributedVector::from_data(alpha.raw_data());
                    let precond = preconditioners::IncompleteLowerUpper::from_matrix(&lhs, 1);
                    let result = solvers::bi_conjugate_gradient_stab(
                        &mut solution, 
                        &lhs, 
                        &rhs, 
                        &precond, 
                        &comm, 
                        parameters.alpha_linear_options,
                    )?;
                    alpha.set_from(solution.data());
                    if rank == 0 { println!("- solved for alpha: {}", result); }
                    compute_gradients(
                        &mut alpha_grad,
                        &alpha,
                        alpha_constraints.as_bc(),
                        &mesh,
                    );
                    compute_limiters::<f64, Vector<DIM>, f64, f64, f64, f64, DIM>(
                        &mut alpha_lim,
                        &alpha,
                        &alpha_grad,
                        schemes::limiters::LimitedLinear(1.0),
                        schemes::faceinterp::Upwind::new(&phi),
                        schemes::facengrad::Corrected::new(&alpha_grad, 1.0),
                        alpha_constraints.as_bc(),
                        &mesh,
                    );
                }

                // update the density field
                for cell in mesh.iter_cells() {
                    let a = alpha[cell.id()].max(0.0).min(1.0);
                    let rho = parameters.gas_density * (1.0 - a) + parameters.fluid_density * a;
                    density[cell.id()] = rho;
                }
                density.update();
            }
            // interpolate rho on faces
            {
                let interp = schemes::faceinterp::LimitedLinear::<f64, f64, DIM>::new(&phi, &alpha_lim);
                for face in mesh.iter_faces() {
                    let (t0, t1, r) = interp.terms(&face, &mesh);
                    match face.neighbor() {
                        FaceNeighbor::Boundary(b) => {
                            density_face[face.id()] = density[face.owner()];
                            // if b == inlet {
                            //     density_face[face.id()] = if face.center()[hdim] < 0.0 {parameters.fluid_density} else {parameters.gas_density};
                            // }
                        },
                        FaceNeighbor::Cell(c1) => {
                            density_face[face.id()] = 
                                t0 * density[face.owner()]
                                + t1 * density[c1]
                                + r;
                        }
                    }
                }
            }
            density_face.update();
            compute_gradients(
                &mut density_grad,
                &density,
                |_| {(1.0, 0.0)},
                &mesh
            );

            // update the surface tension
            let mut div_curvature = alpha_grad.clone();
            for cell in mesh.iter_cells() {
                let d = div_curvature[cell.id()].norm().max(1e-14);
                div_curvature[cell.id()] /= d;
            }
            div_curvature.update();
            let mut div_curvature_face = Field::<f64, geometry::Face, DIM>::from(&mesh);
            let interp = schemes::faceinterp::Linear::<f64, f64, DIM>::new();
            for face in mesh.iter_faces() {
                let (t0, t1, _) = interp.terms(&face, &mesh);
                div_curvature_face[face.id()] = match face.neighbor() {
                    FaceNeighbor::Boundary(_) => div_curvature[face.owner()].dot(face.normal()),
                    FaceNeighbor::Cell(c1) => (div_curvature[face.owner()] * t0 + div_curvature[c1] * t1).dot(face.normal())
                };
            }
            div_curvature_face.update();
            let mut kappa = Field::<f64, geometry::Cell, DIM>::from(&mesh);
            for face in mesh.iter_faces() {
                kappa[face.owner()] += div_curvature_face[face.id()] * face.area();
                match face.neighbor() {
                    FaceNeighbor::Cell(c1) => {kappa[c1] -= div_curvature_face[face.id()] * face.area();}
                    _ => {}
                }
            }
            kappa.update();

            for cell in mesh.iter_cells() {
                let sigma = parameters.surface_tension;
                let kappa = - kappa[cell.id()] / cell.volume();
                let a_grad = alpha_grad[cell.id()];

                surface_tension_source[cell.id()] = sigma * kappa * a_grad;
            }
            surface_tension_source.update();
        }

        time += dt;

        if time_iter % 50 == 0 {
            PvtuWriter::new(&mesh)
                .with("U", &velocity)
                .with("p", &pressure)
                .with("alpha", &alpha)
                .with("rho", &density)
                .write(format!("examples/ex6/data/solution_{}.pvtu", write_iter).as_str())?;
            
            write_iter += 1;
        }

    }

    Ok(())
}




fn main() -> Result<(), Box<dyn std::error::Error>> {

    let universe = mpi::initialize().unwrap();
    let world = universe.world();

    ex6::<3>(
        world,
        Parameters { 
            velocity: 1.0, 
            viscosity: 0.02, 
            fluid_density: 2.0, 
            gas_density: 1.0, 
            surface_tension: 0.05,
            time_step: 1e-4, 
            cfl: 1.0,
            time_iterations: 1000,
            outer_correctors: 1,
            momentum_predictor: false,
            pressure_correctors: 3,
            pressure_linear_options: LinearSolverOptions { 
                relative_tolerance: 0.05, 
                absolute_tolerance: 1e-5, 
                max_iterations: 500,
            },
            pressure_linear_options_final: LinearSolverOptions { 
                relative_tolerance: 1e-5, 
                absolute_tolerance: 1e-5, 
                max_iterations: 500,
            },
            alpha_correctors: 2,
            alpha_linear_options: LinearSolverOptions {
                relative_tolerance: 1e-6,
                absolute_tolerance: 1e-6,
                max_iterations: 500,
            },
        },
    )?;

    Ok(())
}