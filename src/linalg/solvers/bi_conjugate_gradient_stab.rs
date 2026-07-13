use std::ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign};

use mpi::traits::Equivalence;

use crate::{core::{Communicator, mesh::Geometry}, error::Error, linalg::{DistributedMatrix, DistributedVector, Preconditioner, solvers::LinearSolverOptions}};

use super::LinearSolverInfo;



static SOLVER_NAME: &str = "BiConjugateGradientStab";

pub fn bi_conjugate_gradient_stab<'a, G: Geometry<DIM>, LHS, RHS, const DIM: usize>(
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
    LHS: Copy + Clone + Default  + std::ops::Mul<RHS, Output = RHS>
{
    let tolerance = options.absolute_tolerance;
    let relative_tolerance = options.relative_tolerance;
    let max_iter = options.max_iterations;

    let bnorm = comm.single().reduce_add(rhs.dot(&rhs)).sqrt();

    if bnorm < 1e-14 {
        for i in 0..solution.len() {
            solution[i] = RHS::default();
        }
        return Ok(LinearSolverInfo { solver_identifier: SOLVER_NAME, iterations: 0, initial_residual: bnorm, final_residual: bnorm, history: Some(vec![bnorm]) });
    }

    // allocate solver memory
    let mut p = DistributedVector::from_size(solution.len());
    let mut r = DistributedVector::from_size(rhs.len());
    let mut rh = DistributedVector::from_size(rhs.len());
    let mut z = DistributedVector::from_size(solution.len());
    let mut y = DistributedVector::from_size(solution.len());
    let mut v = DistributedVector::from_size(rhs.len());
    let mut h = DistributedVector::from_size(rhs.len());
    let mut s = DistributedVector::from_size(rhs.len());
    let mut t = DistributedVector::from_size(rhs.len());


    // compute residual
    comm.collect(solution.data_mut());
    lhs.imul(&mut r, solution.data());
    r -= rhs;
    r *= -1.0;

    for i in 0..r.len() {
        rh[i] = r[i];
    }

    let initial_residual = comm.single().reduce_add(r.dot(&r)).sqrt() / bnorm;

    if initial_residual < tolerance {
        return Ok(LinearSolverInfo { solver_identifier: SOLVER_NAME, iterations: 0, initial_residual: initial_residual, final_residual: initial_residual, history: Some(vec![initial_residual]) });
    }

    // solve Mz = r
    //preconditioner.precondition(&mut z, &r, comm);
    //comm.communicate(z.data_mut());

    // set p = r
    p.set_smaller(&r.data());
    comm.collect(p.data_mut());


    // rho
    let mut rho = comm.single().reduce_add(rh.dot(&r));


    let mut error = 1.0;
    let mut iter = 0;
    let mut history = Vec::<f64>::new();

    while error > tolerance {
        if iter > max_iter {break;}


        //println!("p = {:?}", p);
        preconditioner.precondition(&mut y, &p, comm);
        comm.collect(y.data_mut());


        lhs.imul(&mut v, y.data());

        let rh_dot_v = comm.single().reduce_add(rh.dot(&v));
        let alpha = rho / rh_dot_v;

        //println!("a = {:.3e}", alpha);

        for i in 0..h.len() {
            h[i] = solution[i] + y[i] * alpha;
            s[i] = r[i] - v[i] * alpha;
        }

        // check h
        let s_dot_s_sqrt = comm.single().reduce_add(s.dot(&s)).sqrt() / bnorm;
        if (s_dot_s_sqrt < tolerance) || (s_dot_s_sqrt / initial_residual < relative_tolerance) {
            error = s_dot_s_sqrt;
            history.push(error);

            for i in 0..h.len() {
                solution[i] = h[i];
            }
            comm.collect(solution.data_mut());
            iter += 1;

            break;
        }
    

        preconditioner.precondition(&mut z, &s, comm);
        comm.collect(z.data_mut());
        
        lhs.imul(&mut t, z.data());

        let t_dot_s = comm.single().reduce_add(t.dot(&s));
        let t_dot_t = comm.single().reduce_add(t.dot(&t));

        let omega = t_dot_s / t_dot_t;

        //println!("w = {:.3e}", omega);

        for i in 0..h.len() {
            solution[i] = h[i] + z[i]* omega;
            r[i] = s[i] - t[i] * omega;
        }
        comm.collect(solution.data_mut());

        let r_dot_r_sqrt = comm.single().reduce_add(r.dot(&r)).sqrt() / bnorm;
        if (r_dot_r_sqrt < tolerance) || (r_dot_r_sqrt / initial_residual < relative_tolerance) {
            error = r_dot_r_sqrt;
            history.push(error);
            iter += 1;
            break;
        }

        // check r

        let rho_new = comm.single().reduce_add(rh.dot(&r));

        let beta = (rho_new / rho) * (alpha / omega);

        rho = rho_new;

        if rho.abs() > (1e-15 * bnorm) {
            for i in 0..r.len() {
                p[i] = r[i] + (p[i] - v[i] * omega) * beta;
            }
            comm.collect(p.data_mut());
        } else {
            // restart
            //if comm.rank() == 0 {println!("  bicgstab warning: restarting at iteration {}, {:.3e}", iter, r_dot_r_sqrt / bnorm);}
            for i in 0..r.len() {
                rh[i] = r[i];
                p[i] = r[i];
            }
            comm.collect(p.data_mut());
            rho = comm.single().reduce_add(rh.dot(&r));
        }

        error = r_dot_r_sqrt / bnorm;
        //println!("{}: error = {:.3e}", iter, error);
        history.push(error);
        iter += 1;
    }


    Ok(LinearSolverInfo { solver_identifier: SOLVER_NAME, iterations: iter, initial_residual, final_residual: error, history: Some(history) })
}

