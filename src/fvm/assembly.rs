


use std::ops::{AddAssign, Mul, Sub, SubAssign};

use crate::{Mesh, fvm::terms::Term, linalg::{DistributedMatrix, DistributedVector}, prelude::{FaceNeighbor, FaceRef}};

use std::fmt::Debug;


pub fn assemble<'a, V, Lhs, Rhs, const DIM: usize>(
    equation: impl Term<DIM, Lhs = Lhs, Rhs = Rhs>,
    boundary_condition: impl Fn(&FaceRef<DIM>) -> (Lhs, V),
    mesh: &'a Mesh<DIM>,
) -> (DistributedMatrix<Lhs>, DistributedVector<Rhs>) 
where 
Lhs: Default + Copy + AddAssign + SubAssign + Mul<Lhs, Output = Lhs> + Mul<Rhs, Output = Rhs> + Mul<V, Output = Rhs> + Mul<f64, Output = Lhs> + Debug,
Rhs: Default + Copy + AddAssign + SubAssign + Mul<f64, Output = Rhs> + Sub<Rhs, Output=Rhs> + Debug,
V: Debug + Copy,
{

    let mut lhs = DistributedMatrix::from_cut_sparsity(mesh.cell_to_cell_sparsity(), mesh.n_cells());
    let mut rhs = DistributedVector::from_size(mesh.n_cells());


    for cell in mesh.iter_cells() {

        let (l, r) = equation.cell_terms(&cell, &mesh);

        lhs[[cell.id(), cell.id()]] = l * cell.volume();
        rhs[cell.id()] = r * cell.volume();
    }


    for face in mesh.iter_faces() {


        let (li, lj, r) = equation.face_terms(&face, &mesh);

        let li = li * face.area();
        let lj = lj * face.area();
        let r = r * face.area();

        let i = face.owner();
        let celli = mesh.cell(i);

        match face.neighbor() {
            FaceNeighbor::Cell(j) => {
                let cellj = mesh.cell(j);

                if celli.owned() {
                    lhs[[i, i]] += li;
                    lhs[[i, j]] += lj;

                    rhs[i] += r;
                }
                if cellj.owned() {
                    lhs[[j, i]] -= li;
                    lhs[[j, j]] -= lj;

                    rhs[j] -= r;
                }
            },
            FaceNeighbor::Boundary(_) => {

                let (bndi, bnde) = boundary_condition(&face);

                // value_j = bndi * value_i + bnde

                lhs[[i, i]] += li;
                lhs[[i, i]] += lj * bndi;
                rhs[i] += r - lj * bnde;
            },
        }

    }

    (lhs, rhs)
}

