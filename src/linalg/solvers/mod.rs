

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


#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize), serde(rename_all = "kebab-case"))]
#[derive(Debug, Clone, Copy)]
pub struct LinearSolverOptions {
    pub relative_tolerance: f64,
    pub absolute_tolerance: f64,
    pub max_iterations: usize,
}

impl std::default::Default for LinearSolverOptions {
    fn default() -> Self {
        Self {
            relative_tolerance: 1e-6,
            absolute_tolerance: 1e-6,
            max_iterations: 1000,
        }
    }
}


impl std::fmt::Display for LinearSolverInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "(solver={}, iterations={}, initial_residual={:.3e}, final_residual={:.3e})", self.solver_identifier, self.iterations, self.initial_residual, self.final_residual)
    }
}