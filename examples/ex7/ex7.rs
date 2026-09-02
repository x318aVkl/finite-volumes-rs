// refine using the p4est interface

use finite_volumes::fvm::assembly::assemble;
use finite_volumes::linalg::solvers::{bi_conjugate_gradient_stab, conjugate_gradient};
use finite_volumes::prelude::*;
use finite_volumes::refine::context::RefinementContext;
use mpi::traits::CommunicatorCollectives;

fn ex7<const DIM: usize>(world: MpiCommunicator,) -> Result<(), finite_volumes::error::Error> {

    let mut refinement: RefinementContext<DIM> = RefinementContext::read(std::fs::File::open("examples/ex7/mesh.su2")?, world.duplicate())?;
    refinement.partition();

    let mut mesh = refinement.mesh()?;

    println!("rank {} base mesh size: {}", world.rank(), mesh.n_cells());


    for level in 1..=6 {
        if world.rank() == 0 {
            println!("=== level {} ===", level);
        }
        world.barrier();
        refinement.refine_uniform();
        refinement.partition();

        if world.rank() == 0 { println!("done with refinement"); }
        world.barrier();
    }

    for level in 1..=1 {
        if world.rank() == 0 {
            println!("=== level {} ===", level);
        }
        world.barrier();
        refinement.refine(|cell| {
            cell.corner(0)[0] < 0.
        });
        refinement.coarsen(|cells| {
            let mut c = 0.;
            for i in 0..cells.len() {
                c += (cells[i].corner(0)[0].powi(2) + cells[i].corner(0)[1].powi(2)).sqrt();
            }
            c /= cells.len() as f64;
            c < 0.5
        });
        refinement.balance();
        refinement.partition();

        if world.rank() == 0 { println!("done with refinement"); }
        world.barrier();

        mesh = refinement.mesh()?;

        // total area
        let mut total_volume = 0.;
        for cell in mesh.iter_cells() {
            total_volume += cell.volume()
        }

        println!("rank {} mesh size: {}, volume: {}", world.rank(), mesh.n_cells(), total_volume);

        world.barrier();

        let mut volume = 0.0;
        world.all_reduce_into(&total_volume, &mut volume, mpi::collective::SystemOperation::sum());
        if world.rank() == 0 {println!("total volume = {}", volume);}

        world.barrier();


        // solve a simple poisson equation on the mesh
        let source = mesh.iter_cells().map(|_cell| 0.0).collect::<Vec<_>>().to_field(&mesh);
        let diffusion = mesh.iter_faces().map(|_face| 0.1).collect::<Vec<_>>().to_field(&mesh);
        let flux = mesh.iter_faces().map(|face| {
            let velocity = Vector::one();
            velocity.dot(face.normal())
        }).collect::<Vec<_>>().to_field(&mesh);
        let comm = Communicator::<geometry::Cell, DIM>::from(&mesh);

        let previous = mesh.iter_cells().map(|cell| {
            if cell.center().x() < 0. {
                1.0
            } else {
                0.
            }
        }).collect::<Vec<_>>().to_field(&mesh);

        let wall = mesh.patch_id("wall").unwrap();
        let bot = mesh.patch_id("bot").unwrap();

        let (lhs, rhs) = assemble(
            terms::source(&source)
            + terms::time(schemes::time::Euler::new(&previous, 0.1))
            + terms::convection(schemes::faceinterp::Upwind::new(&flux), &flux)
            - terms::laplacian(schemes::facengrad::Orthogonal::new(), &diffusion), 
            |face| {
                if face.boundary().unwrap() == wall {
                    (0.0, 0.0)
                } else if face.boundary().unwrap() == bot {
                    (1.0, 0.0)
                } else {
                    panic!("boundary face has patch {:?}", face.boundary())
                }
            }, 
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
        println!("solved poisson equation: {}", result);

        let field: Field<f64, geometry::Cell, DIM> = solution.data().to_vec().to_field(&mesh);

        let rank_field: Field<f64, geometry::Cell, DIM> = mesh.iter_cells().map(|_| world.rank().into()).collect::<Vec<_>>().to_field(&mesh);

        PvtuWriter::new(&mesh)
            .with("u", &field)
            .with("rank", &rank_field)
            .write(format!("examples/ex7/solution_{}.pvtu", level).as_str()).unwrap();

    }

    Ok(())
}


fn main() -> Result<(), finite_volumes::error::Error> {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();

    ex7::<2>(world)
}