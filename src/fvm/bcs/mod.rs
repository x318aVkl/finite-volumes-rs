use std::ops::Mul;

use crate::{Mesh, core::field::PatchField, prelude::{FaceRef, Unit, Zero}};






pub trait BoundaryCondition<const DIM: usize> {
    type Lhs;
    type Rhs;
    fn bc<'a>(&'a self, mesh: &'a Mesh<DIM>) -> impl Fn(&FaceRef<'a, DIM>) -> (Self::Lhs, Self::Rhs);
}



pub enum StandardBoundaryCondition<T, Lhs, const DIM: usize> {
    FixedValue(PatchField<T, DIM>),
    FixedNormalGradient(PatchField<T, DIM>),
    Mixed(PatchField<Lhs, DIM>, PatchField<T, DIM>),
}



impl<T, Lhs, const DIM: usize> BoundaryCondition<DIM> for StandardBoundaryCondition<T, Lhs, DIM> where Lhs: Zero + Unit + Copy, T: Copy + Mul<f64, Output = T> {
    type Lhs = Lhs;
    type Rhs = T;

    fn bc<'a>(&'a self, mesh: &'a Mesh<DIM>) -> impl Fn(&FaceRef<'a, DIM>) -> (Self::Lhs, Self::Rhs) {
        move |face| {
            match self {
                Self::FixedValue(v) => (Lhs::zero(), v[face.id()]),
                Self::FixedNormalGradient(g) => {
                    let gf = g[face.id()];
                    let cell = mesh.cell(face.owner());
                    let dfcn = (face.center() - cell.center()).dot(face.normal());
                    (Lhs::unit(), gf * dfcn)
                },
                Self::Mixed(c, v) => {
                    // boundary_value = C*cell_Value + v
                    (c[face.id()], v[face.id()])
                }
            }
        }
    }

}

