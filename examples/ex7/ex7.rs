// refine using the p4est interface

use finite_volumes::fvm::assembly::assemble;
use finite_volumes::fvm::tools::gradients::compute_gradients;
use finite_volumes::fvm::tools::limiters::compute_limiters;
use finite_volumes::linalg::solvers::{bi_conjugate_gradient_stab};
use finite_volumes::prelude::*;
use finite_volumes::refine::amr::AMRHandler;
use finite_volumes::refine::context::RefinementMesh;
use mpi::traits::CommunicatorCollectives;

fn ex7<const DIM: usize>(world: MpiCommunicator,) -> Result<(), finite_volumes::error::Error> {

    let dt = 0.01;
    let mu = 1e-5;

    let mut refinement: RefinementMesh<DIM> = RefinementMesh::read(std::fs::File::open("examples/ex7/mesh.su2")?, world.duplicate())?;
    refinement.partition();

    let mut mesh = refinement.build_mesh()?;
    println!("rank {} base mesh size: {}", world.rank(), mesh.n_cells());


    for level in 1..=3 {
        if world.rank() == 0 {
            println!("=== level {} ===", level);
        }
        refinement.refine_uniform();
        refinement.partition();

        if world.rank() == 0 { println!("done with refinement"); }
        world.barrier();
    }

    mesh = refinement.build_mesh()?;
    println!("rank {} base mesh size: {}", world.rank(), mesh.n_cells());

    // build the solution field
    let mut field: Field<f64, geometry::Cell, DIM> = mesh.iter_cells().map(|cell| {
        if cell.center().x() < 0. {0.} else {0.}
    }).collect::<Vec<_>>().to_field(&mesh);

    let mut source = mesh.iter_cells().map(|cell| {
        let mut c = Vector::new();
        c[1] += 0.5;
        if (cell.center() - c).norm() < 0.05 {
            1.
        } else {
            0.
        }
    }).collect::<Vec<_>>().to_field(&mesh);
    let mut diffusion = mesh.iter_faces().map(|_face| mu).collect::<Vec<_>>().to_field(&mesh);
    let mut flux = mesh.iter_faces().map(|face| {
        let mut velocity = Vector::one();
        velocity[0] = -face.center()[1];
        velocity[1] = face.center()[0];
        velocity /= velocity.norm();
        velocity.dot(face.normal())
    }).collect::<Vec<_>>().to_field(&mesh);

    let mut comm = Communicator::<geometry::Cell, DIM>::from(&mesh);

    // let wall = mesh.patch_id("wall").unwrap();
    // let bot = mesh.patch_id("bot").unwrap();

    let mut time = 0.;

    let mut write_iter = 0;

    let bc = |flux: &Field<f64, geometry::Face, DIM>| {
        let f = flux.clone();
        move |face: &FaceRef<DIM>| {
            if f[face.id()] < 0.0 {
                // flux enters, inlet
                (0.0, 0.0)
            } else {
                (1.0, 0.0)
            }
        }
    };

    for time_iter in 1..=1000 {
        if world.rank() == 0 {println!("=== iter {}, time {} ===", time_iter, time);}
        // if world.rank() == 0 {
        //     println!("=== level {} ===", level);
        // }
        // refinement.refine(|cell| {
        //     cell.corner(0)[1] < 0.
        // });
        // refinement.coarsen(|cells| {
        //     let mut c = 0.;
        //     for i in 0..cells.len() {
        //         c += ((cells[i].corner(0)[0] + 1.0).powi(2) + cells[i].corner(0)[1].powi(2)).sqrt();
        //     }
        //     c /= cells.len() as f64;
        //     c < 0.5
        // });
        // refinement.balance();
        // refinement.partition();

        // if world.rank() == 0 { println!("done with refinement"); }
        // world.barrier();

        // mesh = refinement.build_mesh()?;

        // total area

        // solve a simple poisson equation on the mesh
        let previous = field.clone();

        let mut grads = Field::from(&mesh);
        compute_gradients(
            &mut grads,
            &field,
            bc(&flux),
            &mesh
        );

        let mut limiter = Field::from(&mesh);
        compute_limiters::<f64, Vector<DIM>, f64, f64, f64, f64, DIM>(
            &mut limiter,
            &field,
            &grads,
            schemes::limiters::LimitedLinear(1.0),
            schemes::faceinterp::Upwind::new(&flux),
            schemes::facengrad::Corrected::new(&grads, 1.0),
            bc(&flux),
            &mesh,
        );

        let (lhs, rhs) = assemble(
            terms::time(schemes::time::Euler::new(&previous, dt))
            + terms::convection(schemes::faceinterp::LimitedLinear::new(&flux, &limiter), &flux)
            - terms::laplacian(schemes::facengrad::Corrected::new(&grads, 1.0), &diffusion)
            - terms::source(&source), 
            bc(&flux),
            &mesh,
        );

        let mut solution = DistributedVector::from_size(mesh.n_total_cells());
        let precond = preconditioners::IncompleteLowerUpper::from_matrix(&lhs, 1);
        let result = bi_conjugate_gradient_stab(
            &mut solution,
            &lhs,
            &rhs,
            &precond,
            &comm,
            LinearSolverOptions::default(),
        )?;
        if world.rank() == 0 {println!("solved poisson equation: {}", result);}

        time += dt;

        field.set_from(solution.data());

        if time_iter % 25 == 0 {
            PvtuWriter::new(&mesh)
                .with("u", &field)
                .write(format!("examples/ex7/data/solution_{}.pvtu", write_iter).as_str()).unwrap();
            write_iter += 1;
        }

        if (time_iter % 3 == 0) || (time_iter < 10) {
            // refine
            let mut grad = Field::<Vector<DIM>, geometry::Cell, DIM>::from(&mesh);

            compute_gradients(&mut grad, &field, bc(&flux), &mesh);

            let max_gradn = mesh.iter_cells().map(|cell| {grad[cell.id()].norm()}).reduce(|a, b| a.max(b)).unwrap();
            let max_gradn = comm.single().reduce_max(max_gradn);
            if world.rank() == 0 {println!("max gradn = {}", max_gradn)};

            let criteria = mesh.iter_cells().map(|cell| {
                (grad[cell.id()].norm() / max_gradn).powf(0.1)
            }).collect::<Vec<_>>().to_field(&mesh);


            (refinement, mesh) = AMRHandler::new(
                refinement,
                mesh,
                criteria
            )
            .with_tolerances(0.5, 0.1)
            .with_levels(6, 1)
            .with_transfer(|t| {
                field = t.transfer_field_linear(field.clone(), &grad)?;

                Ok(())
            })
            .apply()?;

            let new_ncells = comm.single().reduce_add(mesh.n_cells());
            if world.rank() == 0 {println!("number of cells: {}", new_ncells)}


            source = mesh.iter_cells().map(|cell| {
                let mut c = Vector::new();
                c[1] += 0.5;
                if (cell.center() - c).norm() < 0.05 {
                    1.
                } else {
                    0.
                }
            }).collect::<Vec<_>>().to_field(&mesh);
            diffusion = mesh.iter_faces().map(|_face| mu).collect::<Vec<_>>().to_field(&mesh);
            flux = mesh.iter_faces().map(|face| {
                let mut velocity = Vector::one();
                velocity[0] = -face.center()[1];
                velocity[1] = face.center()[0];
                velocity /= velocity.norm();
                velocity.dot(face.normal())
            }).collect::<Vec<_>>().to_field(&mesh);

            comm = Communicator::from(&mesh);

        }
    }

    Ok(())
}


fn main() -> Result<(), finite_volumes::error::Error> {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();

    ex7::<2>(world)
}