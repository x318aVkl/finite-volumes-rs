/*
    Example 4

    Refine a mesh using adaptive mesh refinement

*/

use finite_volumes::{fvm::{assembly::assemble, schemes, terms, tools::{gradients::compute_gradients, limiters::compute_limiters}}, prelude::*, refine::{context::RefinementContext, criteria}};



fn ex4<const DIM: usize>() -> Result<(), finite_volumes::error::Error> {

    // create the mesh
    let mesh: Mesh<DIM> = Mesh::read(std::io::BufReader::new(std::fs::File::open("examples/ex4/mesh.msh").unwrap()), None).unwrap();
    let mut mesh_refinement = RefinementContext::from_mesh(mesh);
    let mut mesh = mesh_refinement.mesh().clone();

    //let point = Vector::unit() * 0.5;

    // for i in 0..2 {
    //     mesh = mesh_refinement
    //         .set_criteria(|cell| {
    //             1.0
    //         })
    //         .set_level(0.5)
    //         .refine();

    //     println!("Level {}, ncells = {}", i, mesh.n_cells());
    // }

    let mut velocity = Vector::zero();
    velocity[1] = 1.0;
    let velocity = velocity;

    let get_bc = || {
        |face: &FaceRef<DIM>| {
                if face.center().x().min(face.center().y()).min(face.center().z()).abs() < 1e-10 {
                    // zero gradient
                    (1.0, 0.0)
                } else {
                    // fixed value of zero
                    let x = face.center().x() - 0.5;
                    let t = (x*100.0).tanh();
                    let fv = t;
                    (0.0, 0.0)
                }
            }
    };

    let mut u = Field::<f64, geometry::Cell, DIM>::from_mesh(&mesh);

    // adaptive mesh refinement loops
    for refinements in 0..20 {

        let mut source = Field::<f64, geometry::Cell, DIM>::from_mesh(&mesh);
        let mut mu = Field::<f64, geometry::Face, DIM>::from_mesh(&mesh);
        let mut phi = Field::<f64, geometry::Face, DIM>::from_mesh(&mesh);

        let mut refcriteria = Field::<f64, geometry::Cell, DIM>::from_mesh(&mesh);

        for cell in mesh.iter_cells() {
            let r = cell.center().norm() / 0.5;
            source[cell.id()] = if (1.0 -r).abs() < 0.1 {10.0} else {0.0};
        }

        for face in mesh.iter_faces() {
            mu[face.id()] = 1.0;
            phi[face.id()] = velocity.dot(face.normal());
        }

        let schemes = DynamicSchemeSet::default()
            .with(SchemeType::FaceNormalGradient, "orthogonal")
            .with(SchemeType::FaceInterpolation, "upwind");

        let (
            lhs, 
            rhs
        ) = assemble::<_, f64, _, _>(
                    terms::source(&source)
                - terms::laplacian(
                schemes.facengrad::<_, _, Vector<DIM>, _>(None),
                &mu,
                )
                // + terms::convection(
                //     schemes.faceinterp(Some(&phi), None),
                //     &phi
                // )
            ,
            get_bc(),    // zero value on all boundaries
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
            LinearSolverOptions::default(),
        ).unwrap();

        println!("solved: {}", result);

        u.set_from(solution.data());

        let mut gradients = Field::from_mesh(&mesh);
        compute_gradients(&mut gradients, &u, get_bc(), &mesh);

        finite_volumes::refine::criteria::compute_hessian_criteria(
            &mut refcriteria,
            &gradients,
            &u,
            &mesh
        );

        // write the mesh
        PvtuWriter::new(&mesh)
            .with("u", &u)
            .with("criteria", &refcriteria)
            .with("source", &source)
            .write(format!("examples/ex4/solution_{}.pvtu", refinements).as_str()).unwrap();

        // refine the mesh
        let old_ncells = mesh.n_cells();
        mesh = mesh_refinement
            .set_criteria(|cell| {
                refcriteria[cell.id()]
            })
            .set_level(0.1)
            .refine();
        
        u = mesh_refinement.map_field(u);

        println!("Level {}, ncells = {}", refinements, mesh.n_cells());
        
        if mesh.n_cells() > 200_000 {
            println!("Max number of cells reached, exiting");
            break;
        }
        if mesh.n_cells() == old_ncells {
            println!("Refinement has converged, exiting");
            break;
        }
    }

    Ok(())
}



fn main() -> Result<(), finite_volumes::error::Error> {
    ex4::<3>()
}