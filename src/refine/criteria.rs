use std::ops::{Mul, Sub};

use mpi::traits::Equivalence;

use crate::{Field, Mesh, linalg::Magnitude, prelude::{CellIndex, geometry}};


pub trait ElemwiseMax {
    fn elemwise_max(self, rhs: Self) -> Self;
}
pub trait ElemwiseFloatMax {
    fn elemwise_fmax(self, rhs: f64) -> Self;
}
pub trait ElemwiseFloatMin {
    fn elemwise_fmin(self, rhs: f64) -> Self;
}
pub trait ElemwiseMin {
    fn elemwise_min(self, rhs: Self) -> Self;
}
pub trait ElemwiseAbs {
    fn elemwise_abs(self) -> Self;
}
pub trait ElemwiseDiv {
    fn elemwise_div(self, rhs: Self) -> Self;
}
pub trait ElemwiseOwnMax {
    fn elemwise_own_max(self) -> f64;
}
pub trait ElemwiseOwnMin {
    fn elemwise_own_min(self) -> f64;
}
pub fn compute_hessian_criteria<H, V, const DIM: usize>(
    criteria: &mut Field<f64, geometry::Cell, DIM>,
    hessians: &Field<H, geometry::Cell, DIM>,
    field: &Field<V, geometry::Cell, DIM>,
    mesh: &Mesh<DIM>,
)
where H: Mul<f64, Output = H> + Magnitude<Output = f64> + Copy,
V: Clone + Copy + ElemwiseMax + ElemwiseMin + ElemwiseOwnMax + Sub<V, Output = V> + Equivalence + Default
{
    let mut average_hessiandx2_mag: f64 = 0.0;
    let mut maxv = field[CellIndex::from(0)];
    let mut minv = field[CellIndex::from(0)];
    for cell in mesh.iter_cells() {
        let c0 = cell.id();

        let h = hessians[c0];
        let dx = cell.volume().powf(1.0/3.0);

        let hdx2 = h * dx * dx;

        let hdx2mag = hdx2.magnitude();

        criteria[c0] = hdx2mag;
        average_hessiandx2_mag = average_hessiandx2_mag.max(hdx2mag);

        let vi = field[c0];
        maxv = maxv.elemwise_max(vi);
        minv = minv.elemwise_min(vi);
    }

    average_hessiandx2_mag = mesh.comm().reduce_max(average_hessiandx2_mag);
    maxv = mesh.comm().reduce_max(maxv);
    minv = mesh.comm().reduce_min(minv);

    let delta = (maxv - minv).elemwise_own_max().max(1e-4);
    //println!("delta = {}, hdx2mag = {}", delta, average_hessiandx2_mag);

    let normalize_factor = average_hessiandx2_mag.max(delta);

    for cell in mesh.iter_cells() {
        let c0 = cell.id();
        criteria[c0] /= normalize_factor;
        criteria[c0] = criteria[c0].powf(0.75);
    }

    criteria.update();
}




impl ElemwiseMax for f64 {
    fn elemwise_max(self, rhs: Self) -> Self {
        self.max(rhs)
    }
}

impl ElemwiseFloatMax for f64 {
    fn elemwise_fmax(self, rhs: f64) -> Self {
        self.max(rhs)
    }
}
impl ElemwiseMin for f64 {
    fn elemwise_min(self, rhs: Self) -> Self {
        self.min(rhs)
    }
}
impl ElemwiseFloatMin for f64 {
    fn elemwise_fmin(self, rhs: f64) -> Self {
        self.min(rhs)
    }
}
impl ElemwiseAbs for f64 {
    fn elemwise_abs(self) -> Self {
        self.abs()
    }
}
impl ElemwiseDiv for f64 {
    fn elemwise_div(self, rhs: Self) -> Self {
        self / rhs
    }
}
impl ElemwiseOwnMax for f64 {
    fn elemwise_own_max(self) -> f64 {
        self
    }
}
impl ElemwiseOwnMin for f64 {
    fn elemwise_own_min(self) -> f64 {
        self
    }
}