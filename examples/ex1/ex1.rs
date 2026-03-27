/*
    Example 1

    Solve the convection-diffusion equation

*/

use finite_volumes::prelude::*;


fn ex1<const DIM: usize>(world: MpiCommunicator) -> Result<(), finite_volumes::error::Error> {

    // create the mesh
    let rank = world.rank() as usize;
    let world_size = world.size() as usize;

    let mut mesh: Mesh<2> = Mesh::read(std::io::BufReader::new(std::fs::File::open(if world.size() == 1 {"examples/ex1/mesh.msh".to_string()} else {format!("examples/ex1/mesh_{}.msh", rank)}.as_str()).unwrap()), Some(world)).unwrap();

    // compute and store the face flux for advection
    let mut flux = Field::<f64, geometry::Face, _>::from_mesh(&mesh);
    for face in mesh.iter_faces() {
        let velocity: Vector<_> = [1.0, 1.0].into();
        flux[face.id()] = face.normal().dot(velocity);
    }
    flux.update();

    // solution field
    let mut field = Field::<f64, geometry::Cell, _>::from_mesh(&mesh);

    let mut gradients = Field::<Vector<2>, geometry::Cell, _>::from_mesh(&mesh);

    let dt = 0.1;

    for time_iter in 1..=100 {
        // assemble poisson equation with source term
        let mut lhs = DistributedMatrix::<f64>::from_cut_sparsity(mesh.cell_to_cell_sparsity(), mesh.n_cells());
        let mut rhs = DistributedVector::<f64>::from_size(mesh.n_cells());

        for cell in mesh.iter_cells() {
            let i = cell.id();

            lhs[[i, i]] += 1.0 / dt * cell.volume();

            rhs[i] += field[i] / dt * cell.volume();

        }
        for face in mesh.iter_faces() {
            let i = face.owner();

            let celli = mesh.cell(i);

            let flux = flux[face.id()];

            let diffusion = 0.005;

            match face.neighbor() {
                FaceNeighbor::Cell(j) => {
                    let cellj = mesh.cell(j);

                    // diffusion
                    let t = - 1.0 * diffusion * 1.0 / (cellj.center() - celli.center()).norm() * face.area();
                    if celli.owned() {
                        lhs[[i, i]] -= t;
                        lhs[[i, j]] += t;
                    }
                    if cellj.owned() {
                        lhs[[j, i]] += t;
                        lhs[[j, j]] -= t;
                    }

                    // convection
                    let u = if flux > 0.0 {i} else {j};
                    let d: finite_volumes::core::mesh::CellIndex = if u == i {j} else {i};
                    let t = - 1.0 * flux * face.area();

                    // compute limiter
                    let delta = cellj.center() - celli.center();
                    let grad_n = (field[cellj.id()] - field[celli.id()]) / delta.norm();
                    
                    let r = if grad_n.abs() > 1e-8 {
                        (2.0*gradients[u].dot(delta) / (delta.norm() * grad_n) - 1.0).max(0.0)
                    } else {
                        1.0
                    };
                    let lim = (1.0*r).min(1.0);

                    if celli.owned() {
                        lhs[[i, u]] -= t * (0.5 * lim + (1.0 - lim));
                        lhs[[i, d]] -= t * 0.5 * lim;
                    }
                    if cellj.owned() {
                        lhs[[j, u]] += t * (0.5 * lim + (1.0 - lim));
                        lhs[[j, d]] += t * 0.5 * lim;
                    }
                },
                FaceNeighbor::Boundary(_b) => {
                    let bv = if face.center().y().abs() > 1e-10 {1.0} else {-1.0};
                    
                    // convection
                    let t = -  1.0 * flux * face.area();
                    if flux < 0.0 {
                        // inlet
                        rhs[i] += t * bv;
                    } else {
                        //outlet
                        lhs[[i, i]] -= t;
                    }
                },
            }
        }

        let comm = Communicator::<geometry::Cell, _>::from_mesh(&mesh);

        // solve
        let mut solution = DistributedVector::from_data(field.raw_data());

        //lhs.enforce_system_diagonal_dominance(&mut rhs, &solution, 1.0);

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

        if rank == 0 {println!("iter {}, solved: {}", time_iter, result);}

        if result.initial_residual < 1e-4 {
            break;
        }

        for i in mesh.iter_cells() {
            field[i.id()] = solution[i.id()];
        }
        field.update();

        // compute and update gradients
        for cell in mesh.iter_cells() {

            let mut grad = field[cell.id()] * cell.own_grad();

            for (f, g) in cell.iter_grad() {
                let face = mesh.face(f);
                match face.neighbor() {
                    FaceNeighbor::Cell(_c) => {
                        let c = face.other_cell(cell.id()).unwrap();
                        grad += field[c] * g;
                    },
                    FaceNeighbor::Boundary(_) => {
                        let bv = if face.center().y().abs() > 1e-10 {1.0} else {-1.0};
                        let flux = flux[face.id()];
                        if flux < 0.0 {
                            // inlet
                            grad += bv * g;
                        } else {
                            // zero grad
                            grad += field[cell.id()] * g;
                        }
                    },
                }
            }

            gradients[cell.id()] = grad;
        }
        gradients.update();

    }

    // write the solution
    PvtuWriter::new(&mesh)
        .with("phi", &field)
        .write("examples/ex1/solution.pvtu")
        .unwrap();


    Ok(())
}


fn main() -> Result<(), finite_volumes::error::Error> {

    let universe = mpi::initialize().ok_or(finite_volumes::error::Error::MpiInitializeFailed)?;
    let world = universe.world();

    ex1::<2>(world)?;

    Ok(())
}


