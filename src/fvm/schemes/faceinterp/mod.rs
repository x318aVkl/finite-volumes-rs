use crate::{Mesh, prelude::FaceRef};


pub mod upwind;
pub mod limitedlinear;

pub use upwind::Upwind;
pub use limitedlinear::LimitedLinear;



pub trait FaceInterpolationScheme<const DIM: usize> {
    type Lhs;
    type Rhs;

    fn terms<'a>(&self, face: &'a FaceRef<'a, DIM>, mesh: &'a Mesh<DIM>) -> (Self::Lhs, Self::Lhs, Self::Rhs);
}
