// refine using the p4est interface

use finite_volumes::fvm::assembly::assemble;
use finite_volumes::fvm::tools::gradients::compute_gradients;
use finite_volumes::linalg::solvers::{bi_conjugate_gradient_stab, conjugate_gradient};
use finite_volumes::{prelude::*, refine};
use finite_volumes::refine::context::{RefinementContext, transfer_field_adapt, transfer_field_partition};
use mpi::traits::CommunicatorCollectives;

fn ex7<const DIM: usize>(world: MpiCommunicator,) -> Result<(), finite_volumes::error::Error> {

    let dt = 0.01;

    let mut refinement: RefinementContext<DIM> = RefinementContext::read(std::fs::File::open("examples/ex7/mesh.su2")?, world.duplicate())?;
    refinement.partition();

    let mut mesh = refinement.mesh()?;

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

    mesh = refinement.mesh()?;
    println!("rank {} base mesh size: {}", world.rank(), mesh.n_cells());

    // build the solution field
    let mut field: Field<f64, geometry::Cell, DIM> = mesh.iter_cells().map(|cell| {
        if cell.center().x() < 0. {0.} else {0.}
    }).collect::<Vec<_>>().to_field(&mesh);

    let mut source = mesh.iter_cells().map(|_cell| 0.0).collect::<Vec<_>>().to_field(&mesh);
    let mut diffusion = mesh.iter_faces().map(|_face| 0.001).collect::<Vec<_>>().to_field(&mesh);
    let mut flux = mesh.iter_faces().map(|face| {
        let velocity = Vector::one();
        velocity.dot(face.normal())
    }).collect::<Vec<_>>().to_field(&mesh);

    let mut comm = Communicator::<geometry::Cell, DIM>::from(&mesh);

    let wall = mesh.patch_id("wall").unwrap();
    let bot = mesh.patch_id("bot").unwrap();

    let mut time = 0.;

    let mut write_iter = 0;

    let bc = |flux: &Field<f64, geometry::Face, DIM>| {
        let f = flux.clone();
        move |face: &FaceRef<DIM>| {
            if f[face.id()] < 0.0 {
                // flux enters, inlet
                (0.0, 1.0)
            } else {
                (1.0, 0.0)
            }
        }
    };

    for time_iter in 1..=100 {
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

        // mesh = refinement.mesh()?;

        // total area

        // solve a simple poisson equation on the mesh
        let previous = field.clone();

        let (lhs, rhs) = assemble(
            terms::source(&source)
            + terms::time(schemes::time::Euler::new(&previous, dt))
            + terms::convection(schemes::faceinterp::Upwind::new(&flux), &flux)
            - terms::laplacian(schemes::facengrad::Orthogonal::new(), &diffusion), 
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

        if time_iter % 10 == 0 {
            PvtuWriter::new(&mesh)
                .with("u", &field)
                .write(format!("examples/ex7/data/solution_{}.pvtu", write_iter).as_str()).unwrap();
            write_iter += 1;
        }

        if time_iter % 3 == 0 {
            // refine
            let mut grad = Field::<Vector<DIM>, geometry::Cell, DIM>::from(&mesh);

            compute_gradients(&mut grad, &field, bc(&flux), &mesh);

            let max_gradn = mesh.iter_cells().map(|cell| {grad[cell.id()].norm()}).reduce(|a, b| a.max(b)).unwrap();
            let max_gradn = comm.single().reduce_max(max_gradn);
            if world.rank() == 0 {println!("max gradn = {}", max_gradn)};

            // transfer fields
            let old_refinement = refinement.clone();
            let old_mesh = mesh.clone();

            let min_level = 2;
            let max_level = 6;

            refinement.refine(|cell| {
                let g = grad[cell.local_id.into()];
                let gn = g.norm() / max_gradn;
                let target_level = (gn * (max_level - min_level) as f64).round() as u8 + min_level;
                cell.level < target_level
            });
            refinement.balance();

            mesh = refinement.mesh()?;

            println!("rank {} old: {}, new: {}", world.rank(), old_mesh.n_cells(), mesh.n_cells());
            

            field = transfer_field_adapt::<_, Vector<DIM>, _>(
                &old_refinement,
                &refinement,
                &old_mesh,
                &mesh,
                field,
                None //Some(&grad),
            )?;

            grad = transfer_field_adapt::<_, Matrix<DIM, DIM>, _>(
                &old_refinement,
                &refinement,
                &old_mesh,
                &mesh,
                grad,
                None,
            )?;

            drop(old_refinement);
            drop(old_mesh);

            let old_refinement = refinement.clone();
            let old_mesh = mesh.clone();

            refinement.coarsen(|cells| {
                let mut ave_grad = Vector::new();
                let level = cells[0].level;
                for c in cells.iter() {
                    ave_grad += grad[c.local_id.into()];
                }
                ave_grad /= cells.len() as f64;
                let gn = ave_grad.norm() / max_gradn;
                let target_level = (gn * (max_level - min_level) as f64).round() as u8 + min_level;
                level > target_level
            });
            refinement.balance();

            mesh = refinement.mesh()?;
            

            field = transfer_field_adapt::<_, Vector<DIM>, _>(
                &old_refinement,
                &refinement,
                &old_mesh,
                &mesh,
                field,
                None,
            )?;

            drop(old_refinement);
            drop(old_mesh);

            // now also repartition
            let old_refinement = refinement.clone();
            refinement.partition();

            mesh = refinement.mesh()?;

            field = transfer_field_partition(
                &old_refinement,
                &refinement,
                &mesh,
                field,
            )?;

            drop(old_refinement);

            source = mesh.iter_cells().map(|_cell| 0.0).collect::<Vec<_>>().to_field(&mesh);
            diffusion = mesh.iter_faces().map(|_face| 0.001).collect::<Vec<_>>().to_field(&mesh);
            flux = mesh.iter_faces().map(|face| {
                let velocity = Vector::one();
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