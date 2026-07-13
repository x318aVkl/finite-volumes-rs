use mpi::traits::Equivalence;

use crate::{core::{Communicator, mesh::Geometry}, error::Error, linalg::{DistributedMatrix, DistributedVector, Preconditioner, solvers::LinearSolverOptions}};

use super::LinearSolverInfo;
use std::ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign};



static SOLVER_NAME: &str = "ConjugateGradient";

pub fn conjugate_gradient<'a, G: Geometry<DIM>, LHS, RHS, const DIM: usize>(
    solution: &mut DistributedVector<RHS>,
    lhs: &DistributedMatrix<LHS>,
    rhs: &DistributedVector<RHS>,
    preconditioner: &impl Preconditioner<RHS>,
    comm: &'a Communicator<G, DIM>,
    options: LinearSolverOptions,
) -> Result<LinearSolverInfo, Error> 
where G::IndexType: From<usize>, crate::Mesh<DIM>: crate::core::mesh::MeshGet<'a, G::IndexType>,
    usize: From<G::IndexType>,
    RHS: Copy + Clone + Default + AddAssign + SubAssign + MulAssign<f64> + Mul<RHS, Output = f64> + Mul<f64, Output = RHS> + Add<RHS, Output=RHS> + Sub<RHS, Output=RHS> + Equivalence,
    LHS: Copy + Clone + Default + std::ops::Mul<RHS, Output = RHS>
{
    let tolerance = options.absolute_tolerance;
    let relative_tolerance = options.relative_tolerance;
    let max_iter = options.max_iterations;

    let bnorm = comm.single().reduce_add(rhs.dot(rhs)).sqrt();

    if bnorm < 1e-14 {
        for i in 0..solution.len() {
            solution[i] = RHS::default();
        }
        return Ok(LinearSolverInfo { solver_identifier: SOLVER_NAME, iterations: 0, initial_residual: bnorm, final_residual: bnorm, history: Some(vec![bnorm]) });
    }

    // allocate solver memory
    let mut p = DistributedVector::from_size(solution.len());
    let mut r = DistributedVector::from_size(rhs.len());
    let mut z = DistributedVector::from_size(solution.len());

    let mut ap = DistributedVector::from_size(rhs.len());


    // compute residual
    comm.collect(solution.data_mut());
    lhs.imul(&mut r, solution.data());
    r -= rhs;
    r *= -1.0;

    let initial_residual = comm.single().reduce_add(r.dot(&r)).sqrt() / bnorm;

    // solve Mz = r
    preconditioner.precondition(&mut z, &r, comm);
    comm.collect(z.data_mut());

    // set p = z
    p.set(&z.data());


    let mut error = 1.0;
    let mut iter = 0;
    let mut history = Vec::<f64>::new();

    while error > tolerance {
        if iter > max_iter {break;}


        lhs.imul(&mut ap, p.data());
        let r_dot_z = comm.single().reduce_add(r.dot_smaller(&z));
        let p_dot_ap = comm.single().reduce_add(p.dot_smaller(&ap));
        
        let alpha = r_dot_z / p_dot_ap;


        for i in 0..ap.len() {
            solution[i] += p[i] * alpha;
            r[i] -= ap[i] * alpha;
        }
        comm.collect(solution.data_mut());

        let r_dot_r = comm.single().reduce_add(r.dot(&r));
        error = r_dot_r.sqrt() / bnorm;
        if (error < tolerance) || (error / initial_residual < relative_tolerance) {
            history.push(error);
            iter += 1;
            break;
        }

        preconditioner.precondition(&mut z, &r, comm);
        comm.collect(z.data_mut());

        let r_dot_z_new = comm.single().reduce_add(r.dot_smaller(&z));

        let beta = r_dot_z_new / r_dot_z;

        for i in 0..ap.len() {
            p[i] = z[i] + p[i] * beta;
        }
        comm.collect(p.data_mut());


        history.push(error);
        iter += 1;

        //println!("iter {}, error {:.3e}", iter, error);
    }


    Ok(LinearSolverInfo { solver_identifier: SOLVER_NAME, iterations: iter, initial_residual, final_residual: error, history: Some(history) })
}

