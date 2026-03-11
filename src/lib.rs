pub mod core;
pub mod linalg;
pub mod fvm;
pub mod post;

pub use core::{
    mesh::Mesh,
    vector::Vector,
    matrix::Matrix,
    field::Field,
};


pub mod error {
    pub use crate::core::error::Error;
}

pub mod prelude {

    pub use super::Vector;
    pub use super::Matrix;
    pub use super::Mesh;
    pub use super::Field;
    pub use super::core::evaluator::Evaluator;

    pub use super::core::mesh::geometry;

}