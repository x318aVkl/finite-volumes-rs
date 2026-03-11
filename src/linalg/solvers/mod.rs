

pub mod conjugate_gradient;
pub mod bi_conjugate_gradient_stab;

pub use conjugate_gradient::conjugate_gradient;
pub use bi_conjugate_gradient_stab::bi_conjugate_gradient_stab;


#[derive(Debug)]
pub struct LinearSolverInfo {
    pub solver_identifier: &'static str,
    pub iterations: usize,
    pub initial_residual: f64,
    pub final_residual: f64,
    pub history: Option<Vec<f64>>,
}


impl std::fmt::Display for LinearSolverInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "(solver={}, iterations={}, initial_residual={:.3e}, final_residual={:.3e})", self.solver_identifier, self.iterations, self.initial_residual, self.final_residual)
    }
}