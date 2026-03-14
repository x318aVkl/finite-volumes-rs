use std::{marker::PhantomData, ops::{Div, Mul}};

use crate::{Field, fvm::schemes::time::TimeScheme, prelude::{Unit, geometry}};



pub struct Euler<'a, T, Lhs, const DIM: usize> {
    previous: &'a Field<T, geometry::Cell, DIM>,
    dt: f64,
    pdl: PhantomData<Lhs>,
}


impl<'a, T, Lhs, const DIM: usize> Euler<'a, T, Lhs, DIM> {
    pub fn new(previous: &'a Field<T, geometry::Cell, DIM>, dt: f64) -> Self {
        Self { previous, dt, pdl: PhantomData }
    }
}


impl<'b, T, Lhs, const DIM: usize> TimeScheme<DIM> for Euler<'b, T, Lhs, DIM> where T: Div<f64, Output = T> + Copy, Lhs: Unit + Mul<f64, Output = Lhs> {
    type Lhs = Lhs;
    type Rhs = T;

    fn terms<'a>(&self, cell: &crate::prelude::CellRef<'a, DIM>, _mesh: &crate::Mesh<DIM>) -> (Self::Lhs, Self::Rhs) {
        (
            Lhs::unit() * (1.0 / self.dt),
            self.previous[cell.id()] / self.dt
        )
    }
}

