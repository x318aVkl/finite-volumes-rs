/*
    Example 4

    Refine a mesh using adaptive mesh refinement

*/

use finite_volumes::{fvm::{assembly::assemble, terms}, prelude::*, refine::context::RefinementContext};



fn ex4<const DIM: usize>() -> Result<(), finite_volumes::error::Error> {

    // create the mesh
    let mut mesh: Mesh<DIM> = Mesh::read(std::io::BufReader::new(std::fs::File::open("examples/ex4/mesh.msh").unwrap()), None).unwrap();

    //let point = Vector::unit() * 0.5;

    for i in 0..4 {
        mesh = RefinementContext::from_mesh(mesh)
            .criteria(|cell| {
                1.0
            })
            .level(0.5)
            .refine()
            .build();
        println!("Level {}, ncells = {}", i, mesh.n_cells());
    }

    // refine the mesh a few times
    for i in 0..2 {
        mesh = RefinementContext::from_mesh(mesh)
            .criteria(|cell| {
                let x = cell.center().x().min(1.0 - cell.center().x());
                let y = cell.center().y().min(1.0 - cell.center().y());
                let z = cell.center().z().min(1.0 - cell.center().z());
                let t = x.min(y).min(z);
                if t < cell.volume().powf(1.0/3.0)*2.0 {
                    1.0
                } else {
                    0.0
                }
            })
            .level(0.5)
            .refine()
            .build();
        println!("Level {}, ncells = {}", i, mesh.n_cells());
    }

    let mut u = Field::<f64, geometry::Cell, DIM>::from_mesh(&mesh);
    let mut source = Field::<f64, geometry::Cell, DIM>::from_mesh(&mesh);
    let mut mu = Field::<f64, geometry::Face, DIM>::from_mesh(&mesh);

    for cell in mesh.iter_cells() {
        source[cell.id()] = 1.0;
    }
    for face in mesh.iter_faces() {
        mu[face.id()] = 1.0;
    }

    let schemes = DynamicSchemeSet::default()
        .with(SchemeType::FaceNormalGradient, "orthogonal")
        .with(SchemeType::FaceInterpolation, "limited-linear");

    let (
        lhs, 
        rhs
    ) = assemble::<_, f64, _, _>(
                terms::source(&source)
            - terms::laplacian(
            schemes.facengrad::<_, _, Vector<DIM>, _>(None),
            &mu,
            )
        ,
        |face| {
            (0.0, 0.0)
        },    // zero value on all boundaries
        &mesh,
    );

    let mut solution = DistributedVector::from_data(u.raw_data());

    let comm = Communicator::<geometry::Cell, _>::from_mesh(&mesh);

    let precond = IncompleteLowerUpper::from_matrix(&lhs, 1);
    let result = solvers::bi_conjugate_gradient_stab(
        &mut solution,
        &lhs,
        &rhs,
        &precond,
        &comm,
        1e-8,
        1000,
    ).unwrap();

    println!("solved: {}", result);

    u.set_from(solution.data());

    // write the mesh
    PvtuWriter::new(&mesh)
        .with("u", &u)
        .write("examples/ex4/solution.pvtu").unwrap();


    Ok(())
}



fn main() -> Result<(), finite_volumes::error::Error> {
    ex4::<3>()
}