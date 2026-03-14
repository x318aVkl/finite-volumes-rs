



use crate::{Mesh, prelude::FaceRef};



pub mod orthogonal;
pub mod corrected;

pub use orthogonal::Orthogonal;
pub use corrected::Corrected;



pub trait FaceNormalGradientScheme<const DIM: usize> {
    type Lhs;
    type Rhs;

    fn terms<'a>(&self, face: &'a FaceRef<'a, DIM>, mesh: &'a Mesh<DIM>) -> (Self::Lhs, Self::Lhs, Self::Rhs);
}
