/*
    Example 5

    Solve the linear elasticity equations

*/

use finite_volumes::{fvm::{assembly::assemble, schemes, terms, tools::{gradients::compute_gradients, limiters::compute_limiters}}, prelude::*, refine::{context::RefinementContext, criteria}};



fn ex5<const DIM: usize>() -> Result<(), finite_volumes::error::Error> {

    // Material properties
    let young_modulus = 1.0;
    let poisson_ratio = 0.3;


    // compute lame parameters
    let lambda = young_modulus * poisson_ratio / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio));
    let mu = young_modulus / (2.0 * (1.0 + poisson_ratio));

    println!("lambda = {:.3e}, mu = {:.3e}", lambda, mu);

    // create the mesh
    let mesh: Mesh<DIM> = Mesh::read(std::io::BufReader::new(std::fs::File::open("examples/ex5/mesh.msh").unwrap()), None).unwrap();

    let mut displacement = Field::<Vector<DIM>, geometry::Cell, DIM>::from_mesh(&mesh);
    let mut displacement_grad = Field::<Matrix<DIM, DIM>, geometry::Cell, DIM>::from_mesh(&mesh);

    let mut source = Field::<Vector<DIM>, geometry::Cell, DIM>::from_mesh(&mesh);
    let mut implicit_diffusion = Field::<f64, geometry::Face, DIM>::from_mesh(&mesh);
    let mut explicit_stress = Field::<Vector<DIM>, geometry::Face, DIM>::from_mesh(&mesh);
    let mut implicit_term = Field::<Vector<DIM>, geometry::Cell, DIM>::from_mesh(&mesh);
    
    for face in mesh.iter_faces() {
        implicit_diffusion[face.id()] = 2.0 * mu + lambda;
    }
    implicit_diffusion.update();

    let get_bc = move |estress: Field<Vector<DIM>, geometry::Face, DIM>| {
        move |face: &FaceRef<DIM>| {
            if face.center().z() < 1e-6 {
                // zero displacement
                (0.0, Vector::zero())
            } else if face.center().y() > 0.99999 {
                // fixed pressure
                let devstress_dot_n = estress[face.id()];
                let gn = - devstress_dot_n / (2.0 * mu + lambda);
                let dx = face.delta();
                let mut p = Vector::zero();
                p[1] = - 1.0;
                p /= 2.0 * mu + lambda;
                (1.0, (gn + p) * dx)
            } else {
                // free traction
                let devstress_dot_n = estress[face.id()];
                let gn = - devstress_dot_n / (2.0 * mu + lambda);
                let dx = face.delta();
                (1.0, gn * dx)
            }
        }
    };

    let comm = Communicator::<geometry::Cell, _>::from_mesh(&mesh);

    let schemes = DynamicSchemeSet::default()
        .with(SchemeType::FaceNormalGradient, "orthogonal")
        .with(SchemeType::FaceInterpolation, "linear");

    for iter in 0..1000 {

        // compute the source term
        for cell in mesh.iter_cells() {
            source[cell.id()] = Vector::zero();
            implicit_term[cell.id()] = Vector::zero();
        }
        source.update();
        // compute the explicit stress
        for face in mesh.iter_faces() {
            let cell0 = mesh.cell(face.owner());
            
            let u_grad0 = displacement_grad[cell0.id()];

            //let t0 = mu * u_grad0.transpose() + Matrix::eye() * lambda * u_grad0.trace() - (mu + lambda) * u_grad0;
            let t0 = (mu + lambda) * (u_grad0 - u_grad0.transpose());

            let t1 = match face.neighbor() {
                FaceNeighbor::Cell(cell1) => {
                    let u_grad1 = displacement_grad[cell1];

                    //let t1 = mu * u_grad1.transpose() + Matrix::eye() * lambda * u_grad1.trace() - (mu + lambda) * u_grad1;
                    let t1 = (mu + lambda) * (u_grad1 - u_grad1.transpose());

                    t1
                },
                FaceNeighbor::Boundary(_) => {
                    t0
                }
            };


            let f = face.linear_factor();
            let tface = 
                ( t0 * (1.0 - f) + t1 * f ).dot(face.normal())
            ;

            explicit_stress[face.id()] = tface;
        }

        // compute the source term, divergence of explicit stress
        for face in mesh.iter_faces() {

            let cell0 = mesh.cell(face.owner());

            let tface = explicit_stress[face.id()];

            source[cell0.id()] -= tface * face.area() / cell0.volume();

            match face.neighbor() {
                FaceNeighbor::Cell(cell1) => {
                    let cell1 = mesh.cell(cell1);
                    source[cell1.id()] += tface * face.area() / cell1.volume();
                },
                _ => {}
            };
        }
        source.update();

        let mut tot = 0.0;
        let mut toti = 0.0;
        for cell in mesh.iter_cells() {
            tot += source[cell.id()].norm().powi(2);
            toti += implicit_term[cell.id()].norm().powi(2);
        }
        tot = tot.sqrt();
        toti = toti.sqrt();

        println!("source norm = {tot}, implicit norm = {toti}");

        let (
            lhs, 
            rhs
        ) = assemble::<_, f64, _, _>(
                terms::source(&source)
                - terms::laplacian(
                schemes.facengrad::<f64, Vector<DIM>, Matrix<DIM, DIM>, DIM>(None),
                    &implicit_diffusion,
                )
            ,
            get_bc(explicit_stress.clone()),    // zero value on all boundaries
            &mesh,
        );
        
        let mut solution = DistributedVector::from_data(displacement.raw_data());

        let precond = IncompleteCholesky::from_matrix(&lhs, 1);
        let result = solvers::conjugate_gradient(
            &mut solution, 
            &lhs, 
            &rhs, 
            &precond, 
            &comm, 
            1e-8, 
            1e-8,
            1000
        ).unwrap();

        println!("iter {}, solved: {}", iter, result);

        displacement.set_from(solution.data());

        finite_volumes::fvm::tools::gradients::compute_gradients(
            &mut displacement_grad, 
            &displacement, 
            get_bc(explicit_stress.clone()), 
            &mesh
        );

        if result.initial_residual < 1e-7 {
            break;
        }

    }


    let mut stress = Field::<Matrix<DIM, DIM>, geometry::Cell, DIM>::from_mesh(&mesh);

    for cell in mesh.iter_cells() {
        let epsilon = displacement_grad[cell.id()];
        let epsilon = 0.5 * (epsilon + epsilon.transpose());

        stress[cell.id()] = 2.0 * mu * epsilon + Matrix::eye() * lambda * epsilon.trace();
    }

    // Done! Save the solution
    PvtuWriter::new(&mesh)
        .with("u", &displacement)
        .with("source", &source)
        .with("implicit", &implicit_term)
        .with("stress", &stress)
        .write("examples/ex5/solution.pvtu")
        .unwrap();

    Ok(())
}

fn main() -> Result<(), finite_volumes::error::Error> {
    ex5::<3>()
}