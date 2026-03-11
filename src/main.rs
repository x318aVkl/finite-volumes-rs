

use finite_volumes::{core::{communicator::Communicator, mesh::FaceNeighbor}, linalg::{self, DistributedMatrix, DistributedVector, preconditioners}, post::PvtuWriter, prelude::*};


use mpi::{topology::SimpleCommunicator, traits::Communicator as MpiCommunicator};


fn test_read_write_decompose() -> Result<(), finite_volumes::error::Error> {

    println!("Reading mesh in su2 format");
    let mesh: Mesh<2> = Mesh::read_su2(
        std::io::BufReader::new(
            std::fs::File::open("data/mesh.su2").unwrap()
            )
            , 
            None
        ).unwrap();
    println!("  Read mesh with {} cells", mesh.n_cells());
    
    println!("Writing mesh");
    mesh.write(std::io::BufWriter::new(std::fs::File::create("data/mesh.msh").unwrap())).unwrap();
    
    println!("Writing mesh partitions");
    for (rank, part) in mesh.decompose(4)?.enumerate() {
        let part = part?;

        println!("  Writing part {} with {} cells", rank, part.n_cells());

        part.write(std::io::BufWriter::new(
            std::fs::File::create(format!("data/mesh_{}.msh", rank)).unwrap()
        )).unwrap();
    }

    println!("Reading mesh in own format");
    let mesh: Mesh<2> = Mesh::read(
        std::io::BufReader::new(
            std::fs::File::open("data/mesh.msh").unwrap()
            )
            , 
            None
        ).unwrap();
    println!("  Read mesh with {} cells", mesh.n_cells());

    Ok(())
}



fn test_field_comm(world: SimpleCommunicator) -> Result<(), finite_volumes::error::Error> {

    let rank = world.rank() as usize;

    let mesh: Mesh<2> = Mesh::read(std::io::BufReader::new(std::fs::File::open(format!("data/mesh_{}.msh", rank).as_str()).unwrap()), Some(world)).unwrap();

    // Create a field with a scalar f64 value in every cell
    let mut field = Field::<Vector<2>, geometry::Cell, _>::from_mesh(&mesh);
    let mut field2 = Field::<f64, geometry::Cell, _>::from_mesh(&mesh);

    for n in mesh.iter_cells() {
        field[n.id()] = n.center();
        field2[n.id()] = n.center().x();
    }

    field.update();
    field2.update();

    // create an evaluator that computes values at cell centers
    let mut evaluator = Evaluator::<geometry::Cell, 2>::new();
    let x = evaluator.register_fn(|element| {
        element.center().x()
    });
    let f = evaluator.register_fn(|element| {
        field[element.id()]
    });
    let ff2 = evaluator.register_fn(|element| {
        field2[element.id()] * field[element.id()]
    });


    for element in mesh.iter_all_cells() {
        evaluator.update(element.id(), &mesh);
        let data = evaluator.data().clone();

        let xval = data.get(x);
        let fval = data.get(f);
        let ff2val = data.get(ff2);

        assert!((xval - element.center().x()).abs() < 1e-14);
        assert!((fval - element.center()).norm() < 1e-14);
        assert!((ff2val - element.center()*element.center().x()).norm() < 1e-14);
    }
    println!("Rank {} passed", rank);

    Ok(())
}





fn test_poisson(world: SimpleCommunicator) -> Result<(), finite_volumes::error::Error> {

    let rank = world.rank() as usize;

    let mesh: Mesh<2> = Mesh::read(std::io::BufReader::new(std::fs::File::open(if world.size() == 1 {"data/mesh.msh".to_string()} else {format!("data/mesh_{}.msh", rank)}.as_str()).unwrap()), Some(world)).unwrap();

    let mut flux = Field::<f64, geometry::Face, _>::from_mesh(&mesh);
    for face in mesh.iter_faces() {
        let y = face.center().y();
        let velocity: Vector<_> = [1.0, 1.0].into();
        //let velocity: Vector<2> = [1.0 - y, 1.0].into();
        flux[face.id()] = face.normal().dot(velocity);
    }
    flux.update();

    let mut field = Field::<Vector<2>, geometry::Cell, _>::from_mesh(&mesh);

    let mut gradients = Field::<Matrix<2, 2>, geometry::Cell, _>::from_mesh(&mesh);

    let dt = 0.02;

    for time_iter in 1..=100 {
        // assemble poisson equation with source term
        let mut lhs = DistributedMatrix::<Matrix<2, 2>>::from_cut_sparsity(mesh.cell_to_cell_sparsity(), mesh.n_cells());
        let mut rhs = DistributedVector::<Vector<2>>::from_size(mesh.n_cells());

        for cell in mesh.iter_cells() {
            let i = cell.id();

            lhs[[i, i]] += Matrix::eye() / dt * cell.volume();

            rhs[i] += field[i] / dt * cell.volume();

            // source term equal to other component
            lhs[[i, i]][[0, 1]] += 1.0 * cell.volume();
            lhs[[i, i]][[1, 0]] -= 1.0 * cell.volume();
        }
        for face in mesh.iter_faces() {
            let i = face.owner();

            let celli = mesh.cell(i);

            //let velocity: Vector<2> = [0.1 + y, 1.0].into();
            //let flux = velocity.dot(face.normal());
            let flux = flux[face.id()];

            let diffusion = 0.0;

            match face.neighbor() {
                FaceNeighbor::Cell(j) => {
                    let cellj = mesh.cell(j);

                    // diffusion
                    let t = - Matrix::eye() * diffusion * 1.0 / (cellj.center() - celli.center()).norm() * face.area();
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
                    let t = - Matrix::eye() * flux * face.area();
                    // compute limiter
                    let delta = cellj.center() - celli.center();
                    let grad_n = (field[cellj.id()] - field[celli.id()]) / delta.norm();
                    // let r = if grad_n.norm() > 1e-8 {
                    //     (2.0*gradients[u].dot(delta)/ (delta.norm() * grad_n) - 1.0).max(0.0)
                    // } else {
                    //     1.0
                    // };
                    let r = if grad_n.norm() > 1e-8 {
                        (2.0*grad_n.dot(gradients[u].dot(delta)) / (delta.norm() * grad_n.dot(grad_n)) - 1.0).max(0.0)
                    } else {
                        1.0
                    };
                    let lim = (1.0*r).min(1.0);
                    //let lim = (r*r + r)/(r*r + 1.0);

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
                    let bv = if face.center().y().abs() > 1e-10 {1.0} else {-1.0} * Vector::one();
                    // diffusion
                    //let t = - Matrix::eye() * diffusion * 1.0 / (face.center() - celli.center()).norm() * face.area();
                    //lhs[[i, i]] -= t;
                    //rhs[i] += bv * t;

                    // convection
                    let t = -  Matrix::eye() * flux * face.area();
                    if flux < 0.0 {
                        // inlet
                        rhs[i] += t * bv;
                    } else {
                        //outlet
                        lhs[[i, i]] -= t;
                    }
                },
                FaceNeighbor::None => {
                    panic!("Face neighbor is none")
                }
            }
        }

        let comm = Communicator::<geometry::Cell, _>::from_mesh(&mesh);

        // solve
        let mut solution = DistributedVector::from_data(field.raw_data());

        //lhs.enforce_system_diagonal_dominance(&mut rhs, &solution, 1.0);

        let precond = preconditioners::IncompleteLowerUpper::from_matrix(&lhs, 1);
        let result = linalg::solvers::bi_conjugate_gradient_stab(
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

            let mut grad = field[cell.id()].outer(cell.own_grad());

            for (f, g) in cell.iter_grad() {
                let face = mesh.face(f);
                match face.neighbor() {
                    FaceNeighbor::Cell(c) => {
                        let c = face.other_cell(cell.id()).unwrap();
                        grad += field[c].outer(g);
                    },
                    FaceNeighbor::Boundary(_) => {
                        let bv = if face.center().y().abs() > 1e-10 {1.0} else {-1.0} * Vector::one();
                        let flux = flux[face.id()];
                        if flux < 0.0 {
                            // inlet
                            grad += bv.outer(g);
                        } else {
                            // zero grad
                            grad += field[cell.id()].outer(g);
                        }
                    },
                    _ => panic!(""),
                }
            }

            gradients[cell.id()] = grad;
        }
        gradients.update();

    }

    // write the solution
    PvtuWriter::new(&mesh)
        .with_vector("phi", &field)
        .write("data/test.pvtu")
        .unwrap();

    Ok(())
}





fn main() -> Result<(), finite_volumes::error::Error> {

    //test_read_write_decompose()?;

    let universe = mpi::initialize().ok_or(finite_volumes::error::Error::MpiInitializeFailed)?;
    let world = universe.world();

    if world.size() == 1  {
        test_read_write_decompose()?;
        //test_assembly()?;
    } else {
        //test_convection(world)?;
        //test_diffusion(world)?;
        //test_field_comm(world)?;
        test_poisson(world)?;
    }
    
    Ok(())
}
