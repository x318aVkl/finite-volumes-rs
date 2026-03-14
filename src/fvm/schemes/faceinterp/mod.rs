use crate::{Mesh, prelude::FaceRef};


pub mod upwind;
pub mod limitedlinear;
pub mod linear;

pub use upwind::Upwind;
pub use limitedlinear::LimitedLinear;
pub use linear::Linear;


pub trait FaceInterpolationScheme<const DIM: usize> {
    type Lhs;
    type Rhs;

    fn terms<'a>(&self, face: &'a FaceRef<'a, DIM>, mesh: &'a Mesh<DIM>) -> (Self::Lhs, Self::Lhs, Self::Rhs);
}



impl<'b, Lhs, Rhs, const DIM: usize> FaceInterpolationScheme<DIM> for Box<dyn FaceInterpolationScheme<DIM, Lhs=Lhs, Rhs=Rhs> + 'b> {
    type Lhs = Lhs;
    type Rhs = Rhs;

    fn terms<'a>(&self, face: &'a FaceRef<'a, DIM>, mesh: &'a Mesh<DIM>) -> (Self::Lhs, Self::Lhs, Self::Rhs) {
        self.as_ref().terms(face, mesh)
    }
}

