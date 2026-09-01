// refine using the p4est interface

use finite_volumes::fvm::assembly::assemble;
use finite_volumes::linalg::solvers::conjugate_gradient;
use finite_volumes::prelude::*;
use finite_volumes::refine::context::RefinementContext;
use mpi::traits::CommunicatorCollectives;

fn ex7<const DIM: usize>(world: MpiCommunicator,) -> Result<(), finite_volumes::error::Error> {

    // init the refinement context
    // this initializes the p4est library and supresses outputs from its automated log thing
    finite_volumes::refine::context::initialize(&world);

    let mut refinement: RefinementContext<DIM> = RefinementContext::read(std::fs::File::open("examples/ex7/mesh.su2")?, world.duplicate())?;
    refinement.partition();

    let mut mesh = refinement.mesh()?;

    println!("rank {} base mesh size: {}", world.rank(), mesh.n_cells());


    for cell in mesh.iter_cells() {
        println!("cell faces = {:?}", cell.faces());
        for face in cell.faces() {
            let face = mesh.face(*face);

            println!("    {:?}", face.nodes());
            print!("    ");
            for n in face.nodes() {
                print!("{:?}", mesh.node(*n).position().round(2));
            }
            print!("\n");
        }
    }

    for level in 1..=2 {
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
        refinement.refine_uniform();
        refinement.partition();

        if world.rank() == 0 { println!("done with refinement"); }
        world.barrier();

        mesh = refinement.mesh()?;

        for face in mesh.iter_faces() {
            let faces = face.nodes();
            if world.rank() == 0 {println!("rank {} face {} nodes: {:?} center {:?}", world.rank(), face.id(), faces, face.center());}
        }

        // total area
        let mut total_volume = 0.;
        for cell in mesh.iter_cells() {
            if cell.volume().is_nan() {
                let mut cf= vec![];
                for f in cell.faces() {
                    cf.push(mesh.face(*f).nodes().iter().map(|node| *node).collect::<Vec<_>>());
                }
                println!("cell faces: {:?}", cf);
            }
            total_volume += cell.volume()
        }

        println!("rank {} mesh size: {}, volume: {}", world.rank(), mesh.n_cells(), total_volume);

        world.barrier();

        let mut volume = 0.0;
        world.all_reduce_into(&total_volume, &mut volume, mpi::collective::SystemOperation::sum());
        if world.rank() == 0 {println!("total volume = {}", volume);}

        world.barrier();


        // solve a simple poisson equation on the mesh
        let source = mesh.iter_cells().map(|_cell| -1.0).collect::<Vec<_>>().to_field(&mesh);
        let diffusion = mesh.iter_faces().map(|_face| 1.0).collect::<Vec<_>>().to_field(&mesh);
        let comm = Communicator::<geometry::Cell, DIM>::from(&mesh);

        let (lhs, rhs) = assemble(
            terms::source(&source)
            - terms::laplacian(schemes::facengrad::Orthogonal::new(), &diffusion), 
            |_bnd| {
                (0.0, 0.0)
            }, 
            &mesh,
        );

        let mut solution = DistributedVector::from_size(mesh.n_total_cells());
        let precond = preconditioners::IncompleteCholesky::from_matrix(&lhs, 1);
        let result = conjugate_gradient(
            &mut solution,
            &lhs,
            &rhs,
            &precond,
            &comm,
            LinearSolverOptions::default(),
        )?;
        println!("solved poisson equation: {}", result);

        let field: Field<f64, geometry::Cell, DIM> = solution.data().to_vec().to_field(&mesh);

        PvtuWriter::new(&mesh)
            .with("u", &field)
            .write(format!("examples/ex7/solution_{}.pvtu", level).as_str()).unwrap();

    }

    Ok(())
}


fn main() -> Result<(), finite_volumes::error::Error> {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();

    ex7::<2>(world)
}