/*

    Example 3

    solve a convection-diffusion-reaction equation using the simplified
    assembly functions part of the library with premade terms

*/

use finite_volumes::fvm::{self, schemes, terms};
use finite_volumes::prelude::*;

use finite_volumes::fvm::assembly::assemble;


fn ex3(world: MpiCommunicator) -> Result<(), finite_volumes::error::Error> {

    let rank = world.rank();

    let mesh: Mesh<2> = Mesh::read(std::io::BufReader::new(std::fs::File::open(if world.size() == 1 {"examples/ex3/mesh.msh".to_string()} else {format!("examples/ex3/mesh_{}.msh", rank)}.as_str()).unwrap()), Some(world)).unwrap();

    let mut u = Field::<Vector<4>, geometry::Cell, _>::from_mesh(&mesh);

    let mut mu = Field::<f64, geometry::Face, _>::from_mesh(&mesh);
    let mut phi = Field::<f64, geometry::Face, 2>::from_mesh(&mesh);

    let mut coupled_source = Field::<Matrix<4, 4>, geometry::Cell, _>::from_mesh(&mesh);

    for face in mesh.iter_faces() {
        mu[face.id()] = 0.005;
        phi[face.id()] = face.normal().dot([1.0, 1.0].into());
    }

    for cell in mesh.iter_cells() {
        u[cell.id()] = [0.0, 0.0, 1.0, 1.0].into();
    }
    u.update();

    mu.update();
    phi.update();
    coupled_source.update();

    let mut u_grad = Field::<Matrix<4, 2>, geometry::Cell, _>::from_mesh(&mesh);
    let mut u_lim = Field::<f64, geometry::Face, _>::from_mesh(&mesh);

    let dt = 0.02;

    let comm = Communicator::<geometry::Cell, _>::from_mesh(&mesh);


    // setup the schemes dynamically

    let schemes = DynamicSchemeSet::default()
        .with(SchemeType::FaceInterpolation, "limited-linear");


    fn boundary_condition() -> impl Fn(&FaceRef<'_, 2>) -> (Matrix<4, 4>, Vector<4>) {
        |face| {
        if face.center().y().abs() < 1e-10 {
            (Matrix::zero(), [1.0, 0.0, 0.0, 1.0].into())
        } else if face.center().x().abs() < 1e-10 {
            (Matrix::zero(), [0.0, 1.0, 0.0, 1.0].into())
        } else {
            (Matrix::unit(), Vector::zero())
        }}
    }

    for time_iter in 0..100 {

        let u_old = u.clone();

        // update the coupled source terms
        for cell in mesh.iter_cells() {
            let u = u[cell.id()];
            let t = u[3];
            let h = 20.0;
            let rr = 40.0 * (-4.0/t).exp();
            coupled_source[cell.id()] = [
                [-rr*u[1], 0.0, 0.0, 0.0], 
                [0.0, -rr*u[0], 0.0, 0.0], 
                [rr*u[1], rr*u[0], 0.0, 0.0],
                [h*rr*u[1], h*rr*u[0], 0.0, 0.0]
            ].into();
        }
        coupled_source.update();

        {
            // assemble the convection-diffusion problem
            let (
                lhs, 
                rhs
            ) = assemble::<_, _, _, _>(
                    terms::time(
                        schemes.time(dt, Some(&u_old))
                    )
                    -  terms::linear_source(&coupled_source)
                    + terms::convection(
                        schemes.faceinterp(Some(&phi), Some(&u_lim)),
                    &phi,
                    )
                    - terms::laplacian(
                    schemes.facengrad(Some(&u_grad)),
                    &mu,
                    )
                ,
                boundary_condition(),    // zero value on all boundaries
                &mesh,
            );

            // solve
            let mut solution = DistributedVector::from_data(u.raw_data());

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

            println!("iter {}, solved: {}", time_iter, result);

            u.set_from(solution.data());

        }

        // update the gradients and limiters
        fvm::tools::gradients::compute_gradients(
            &mut u_grad, 
            &mut u, 
            boundary_condition(), 
            &mesh
        );

        fvm::tools::limiters::compute_limiters(
            &mut u_lim, 
            &u, 
            &u_grad, 
            schemes::limiters::LimitedLinear(0.5),
            schemes::faceinterp::Upwind::<_, f64, 2>::new(&phi),
            schemes::facengrad::Orthogonal::<_, f64>::new(), 
            boundary_condition(), 
            &mesh
        );

    }

    // Done! Save the solution
    PvtuWriter::new(&mesh)
        .with("u", &u)
        .write("examples/ex3/solution.pvtu")
        .unwrap();

    Ok(())
}

fn main() -> Result<(), finite_volumes::error::Error> {

    let universe = mpi::initialize().ok_or(finite_volumes::error::Error::MpiInitializeFailed)?;
    let world = universe.world();

    ex3(world)?;

    Ok(())
}


