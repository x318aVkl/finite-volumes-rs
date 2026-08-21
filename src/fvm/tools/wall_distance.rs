use std::collections::HashSet;

use crate::{Field, Mesh, Vector, core::{Communicator, mesh::{FaceNeighbor, FaceRef, PatchIndex, geometry}}, fvm::{assembly::assemble, schemes::{self, faceinterp::FaceInterpolationScheme}, terms, tools::gradients::compute_gradients}, linalg::{DistributedVector, preconditioners, solvers::{LinearSolverOptions, conjugate_gradient}}};




pub fn compute_wall_distance<const DIM: usize>(
    wall_distance: &mut Field<f64, geometry::Cell, DIM>,
    mesh: &Mesh<DIM>,
    wall_patches: Option<Vec<PatchIndex>>,
    max_iterations: usize,
    tolerance: f64,
    solver_options: LinearSolverOptions,
) -> Result<(), crate::error::Error> {

    let wall_patches: HashSet<PatchIndex> = if let Some(wp) = wall_patches {
        wp.into_iter().collect()
    } else {
        let mut patches = HashSet::new();
        for patch in mesh.iter_patch() {
            if patch.name().contains("wall") && !patch.name().contains("slip") {
                patches.insert(patch.id());
            }
        }
        patches
    };

    // now, poisson equation solve
    let p = 3;
    let mut distance_grad = Field::<Vector<DIM>, geometry::Cell, DIM>::from(mesh);
    let mut viscosity = Field::<f64, geometry::Face, DIM>::from(mesh);
    
    for face in mesh.iter_faces() {
        viscosity[face.id()] = 1.0;
    }
    viscosity.update();

    let mut source = Field::<f64, geometry::Cell, DIM>::from(mesh);
    for cell in mesh.iter_cells() {
        source[cell.id()] = - 1.0;
    }

    let bc = |face: &FaceRef<DIM>| {
        match face.boundary() {
            Some(b) => {
                if wall_patches.contains(&b) {
                    (0.0, 0.0)
                } else {
                    (1.0, 0.0)
                }
            },
            None => (1.0, 0.0)
        }
    };
    let comm = Communicator::<geometry::Cell, DIM>::from(mesh);
    let rank = comm.rank();

    for iter in 1..=max_iterations {

        let (lhs, rhs) = assemble(
            - terms::laplacian(schemes::facengrad::Corrected::new(&distance_grad, 1.0), &viscosity)
            + terms::source(&source),
            bc,
            &mesh
        );

        let mut solution = DistributedVector::from_data(wall_distance.raw_data());
        let precond = preconditioners::IncompleteCholesky::from_matrix(&lhs, 1);
        let result = conjugate_gradient(
            &mut solution,
            &lhs,
            &rhs,
            &precond,
            &comm,
            solver_options,
        )?;
        if rank == 0 { println!("- [{}] solved for wall distance: {}", iter, result); }
        wall_distance.set_from(solution.data());
        if result.initial_residual < tolerance {break;}

        // update the gradient and the diffusivity
        compute_gradients(
            &mut distance_grad,
            &wall_distance,
            bc,
            &mesh
        );

        let gradinterp = schemes::faceinterp::Linear::<f64, f64, DIM>::new();

        for face in mesh.iter_faces() {
            let g0 = distance_grad[face.owner()];
            let gf = match face.neighbor() {
                FaceNeighbor::Boundary(_) => g0,
                FaceNeighbor::Cell(c1) => {
                    let g1 = distance_grad[c1];
                    let (t0, t1, _) = gradinterp.terms(&face, &mesh);
                    t0 * g0 + t1 * g1
                }
            };
            viscosity[face.id()] = gf.norm().powi(p - 2);
        }

    }

    // finally, normalize the distance
    for cell in mesh.iter_cells() {

        let gnorm = distance_grad[cell.id()].norm();
        let u = wall_distance[cell.id()];
        let pf = p as f64;

        let v = - gnorm.powi(p - 1) + ((pf / (pf - 1.)) * u + gnorm.powi(p)).powf((pf - 1.)/pf);
        wall_distance[cell.id()] = v;
    }

    Ok(())
}

