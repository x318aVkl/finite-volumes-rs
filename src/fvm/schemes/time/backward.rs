use std::{marker::PhantomData, ops::{Div, Mul, Sub}};

use crate::{Field, fvm::schemes::time::TimeScheme, prelude::{Unit, geometry}};




pub struct Backward<'a, T, Lhs, const DIM: usize> {
    previous: &'a Field<T, geometry::Cell, DIM>,
    previous_2: &'a Field<T, geometry::Cell, DIM>,
    densities: Option<[&'a Field<f64, geometry::Cell, DIM>; 3]>,
    dt: f64,
    a_next: f64,
    a_last: f64,
    a_last_2: f64,
    pdl: PhantomData<Lhs>,
}



impl<'a, T, Lhs, const DIM: usize> Backward<'a, T, Lhs, DIM> {
    pub fn new(previous: &'a Field<T, geometry::Cell, DIM>, previous_2: &'a Field<T, geometry::Cell, DIM>, dt: f64, dt_2: f64) -> Self {
        let wn = dt / dt_2;
        let a_next = (1.0 + 2.0 * wn) / (1.0 + wn);
        let a_last = (1.0 + wn).powi(2) / (1.0 + wn);
        let a_last_2 = wn.powi(2) / (1.0 + wn);
        Self { previous, previous_2, densities: None, dt, a_next, a_last, a_last_2, pdl: PhantomData }
    }
    pub fn new_with_density(previous: &'a Field<T, geometry::Cell, DIM>, previous_2: &'a Field<T, geometry::Cell, DIM>, dt: f64, dt_2: f64, density: &'a Field<f64, geometry::Cell, DIM>,  density_last: &'a Field<f64, geometry::Cell, DIM>,  density_last2: &'a Field<f64, geometry::Cell, DIM>,  ) -> Self {
        let wn = dt / dt_2;
        let a_next = (1.0 + 2.0 * wn) / (1.0 + wn);
        let a_last = (1.0 + wn).powi(2) / (1.0 + wn);
        let a_last_2 = wn.powi(2) / (1.0 + wn);
        Self { previous, previous_2, densities: Some([density, density_last, density_last2]), dt, a_next, a_last, a_last_2, pdl: PhantomData }
    }
}



impl<'b, T, Lhs, const DIM: usize> TimeScheme<DIM> for Backward<'b, T, Lhs, DIM> where T: Div<f64, Output = T> + Mul<f64, Output = T> + Sub<T, Output=T> + Copy, Lhs: Unit + Mul<f64, Output = Lhs> {
    type Lhs = Lhs;
    type Rhs = T;

    fn terms<'a>(&self, cell: &crate::prelude::CellRef<'a, DIM>, _mesh: &crate::Mesh<DIM>) -> (Self::Lhs, Self::Rhs) {
        let (rho0, rho1, rho2) = match self.densities {
            Some(d) => {
                (
                    d[0][cell.id()],
                    d[1][cell.id()],
                    d[2][cell.id()],
                )
            },
            None => {
                (1.0, 1.0, 1.0)
            }
        };
        (
            Lhs::unit() * (self.a_next / self.dt) * rho0,
            (self.previous[cell.id()] * self.a_last * rho1 - self.previous_2[cell.id()] * self.a_last_2 * rho2) / self.dt
        )
    }
}
