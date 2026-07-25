use std::{marker::PhantomData, ops::{Div, Mul}};

use crate::{Field, fvm::schemes::time::TimeScheme, prelude::{Unit, geometry}};



pub struct Euler<'a, T, Lhs, const DIM: usize> {
    previous: &'a Field<T, geometry::Cell, DIM>,
    dt: f64,
    pdl: PhantomData<Lhs>,
    density: Option<[&'a Field<f64, geometry::Cell, DIM>; 2]>,
}


impl<'a, T, Lhs, const DIM: usize> Euler<'a, T, Lhs, DIM> {
    pub fn new(previous: &'a Field<T, geometry::Cell, DIM>, dt: f64) -> Self {
        Self { previous, dt, pdl: PhantomData, density: None, }
    }
    pub fn new_with_density(previous: &'a Field<T, geometry::Cell, DIM>, dt: f64, density: &'a Field<f64, geometry::Cell, DIM>, density_previous: &'a Field<f64, geometry::Cell, DIM>) -> Self {
        Self { previous, dt, pdl: PhantomData, density: Some([density, density_previous]), }
    }
}


impl<'b, T, Lhs, const DIM: usize> TimeScheme<DIM> for Euler<'b, T, Lhs, DIM> where T: Div<f64, Output = T> + Mul<f64, Output = T> + Copy, Lhs: Unit + Mul<f64, Output = Lhs> {
    type Lhs = Lhs;
    type Rhs = T;

    fn terms<'a>(&self, cell: &crate::prelude::CellRef<'a, DIM>, _mesh: &crate::Mesh<DIM>) -> (Self::Lhs, Self::Rhs) {
        let (r0, r1) = match self.density {
            Some(d) => (d[0][cell.id()], d[1][cell.id()]),
            None => (1.0, 1.0),
        };
        (
            Lhs::unit() * (1.0 / self.dt) * r0,
            self.previous[cell.id()] / self.dt * r1
        )
    }
}

