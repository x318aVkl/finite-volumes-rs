use std::ops::{Add, AddAssign, Mul};

use mpi::traits::Equivalence;

use crate::{Field, Matrix, Mesh, Vector, prelude::{FaceNeighbor, FaceRef, geometry}};



pub trait GradientFrom<const DIM: usize> {
    type GradientType;

    fn mul_to_gradient(self, g: Vector<DIM>) -> Self::GradientType;
}



impl<const DIM: usize> GradientFrom<DIM> for f64 {
    type GradientType = Vector<DIM>;

    fn mul_to_gradient(self, g: Vector<DIM>) -> Self::GradientType {
        self * g
    }
}

impl<const N: usize, const DIM: usize> GradientFrom<DIM> for Vector<N> {
    type GradientType = Matrix<N, DIM>;

    fn mul_to_gradient(self, g: Vector<DIM>) -> Self::GradientType {
        self.outer(g)
    }
}


pub fn compute_gradients<'a, V, Lhs, const DIM: usize>(
    gradients: &mut Field<V::GradientType, geometry::Cell, DIM>,
    values: &Field<V, geometry::Cell, DIM>,
    boundary_condition: impl Fn(&FaceRef<'a, DIM>) -> (Lhs, V),
    mesh: &'a Mesh<DIM>,
)
where 
V: GradientFrom<DIM> + Copy + Add<V, Output = V>,
<V as GradientFrom<DIM>>::GradientType: AddAssign + Clone + Default + Equivalence,
Lhs: Mul<V, Output = V>,
{

    for cell in mesh.iter_cells() {

        let mut grad = values[cell.id()].mul_to_gradient(cell.own_grad());

        for (f, g) in cell.iter_grad() {
            let face = mesh.face(f);
            match face.neighbor() {
                FaceNeighbor::Cell(_c) => {
                    let c = face.other_cell(cell.id()).unwrap();
                    grad += values[c].mul_to_gradient(g);
                },
                FaceNeighbor::Boundary(_) => {
                    let (bci, bce) = boundary_condition(&face);

                    let fval = bci * values[cell.id()] + bce;

                    grad += fval.mul_to_gradient(g);
                },
            }
        }

        gradients[cell.id()] = grad;
    }

    gradients.update();

}



