/*

    hbya

    tools for fluid flow problems with pressure-based solution process

*/

use std::ops::Mul;

use crate::{Field, Mesh, Vector, fvm::schemes::{faceinterp::FaceInterpolationScheme, facengrad::FaceNormalGradientScheme}, linalg::{DistributedMatrix, DistributedVector}, prelude::{CellIndex, FaceNeighbor, FaceRef, geometry}};



pub fn compute_hbya_ainv<const DIM: usize>(
    hbya: &mut Field<Vector<DIM>, geometry::Cell, DIM>,
    ainv: &mut Field<f64, geometry::Cell, DIM>,
    velocity: &Field<Vector<DIM>, geometry::Cell, DIM>,
    mlhs: &DistributedMatrix<f64>,
    mrhs: &DistributedVector<Vector<DIM>>,
    mesh: &Mesh<DIM>,
) {

    // compute ainv
    for i in 0..mlhs.nrows() {
        let ci = CellIndex::from(i);

        let cell = mesh.cell(ci);

        ainv[ci] = cell.volume() / mlhs[[i, i]];
    }
    ainv.update();

    // compute hbya
    for i in 0..mrhs.len() {
        let ci = CellIndex::from(i);

        let mut hi = mrhs[i];

        for (j, aij) in mlhs.iter_row(i) {
            if i != j {
                hi -= velocity[CellIndex::from(j)] * aij;
            }
        }

        hbya[ci] = hi / mlhs[[i, i]];
    }
    hbya.update();

}





pub fn intepolate_hbya_ainv_faces<'a, Lhs, const DIM: usize>(
    hbyan_face: &mut Field<f64, geometry::Face, DIM>,
    ainv_face: &mut Field<f64, geometry::Face, DIM>,
    hbya: &Field<Vector<DIM>, geometry::Cell, DIM>,
    ainv: &Field<f64, geometry::Cell, DIM>,
    velocity_bc: impl Fn(&FaceRef<'a, DIM>) -> (f64, Vector<DIM>),
    pressure_bc: impl Fn(&FaceRef<'a, DIM>) -> (f64, f64),
    hbya_interp_scheme: impl FaceInterpolationScheme<DIM, Lhs=Lhs, Rhs=Vector<DIM>>,
    ainv_interp_scheme: impl FaceInterpolationScheme<DIM, Lhs=f64, Rhs=f64>,
    mesh: &'a Mesh<DIM>,
) 
where Lhs: Mul<Vector<DIM>, Output = Vector<DIM>>,
{
    for face in mesh.iter_faces() {
        
        let i = face.owner();

        let (hbya_interpi, hbya_interpj, hbya_interprhs) = hbya_interp_scheme.terms(&face, mesh);
        let (ainv_interpi, ainv_interpj, ainv_rhs) = ainv_interp_scheme.terms(&face, mesh);
        
        match face.neighbor() {
            FaceNeighbor::Cell(j) => {
                hbyan_face[face.id()] = (hbya_interpi*hbya[i] + hbya_interpj*hbya[j] + hbya_interprhs).dot(face.normal());
                ainv_face[face.id()] = ainv_interpi*ainv[i] + ainv_interpj*ainv[j] + ainv_rhs;
            },
            FaceNeighbor::Boundary(_) => {
                //get velocity and pressure bcs
                let (blhsu, bvu) = velocity_bc(&face);
                let (blhsp, bvp) = pressure_bc(&face);

                let p_wall = ((blhsp - 1.0).abs() < 1e-10) && (bvp.abs() < 1e-10);
                let u_wall = (blhsu.abs() < 1e-10) && (bvu.dot(face.normal()).abs() < 1e-10);

                if p_wall && u_wall {
                    // zero since wall
                    hbyan_face[face.id()] = 0.0;
                } else {
                    hbyan_face[face.id()] = hbya[i].dot(face.normal());
                }

                ainv_face[face.id()] = ainv[i];
            },
        }
    }
    hbyan_face.update();
    ainv_face.update();
}





pub fn correct_phi<'a, const DIM: usize>(
    phi: &mut Field<f64, geometry::Face, DIM>,
    hbyan_face: &Field<f64, geometry::Face, DIM>,
    ainv_face: &Field<f64, geometry::Face, DIM>,
    pressure: &Field<f64, geometry::Cell, DIM>,
    fngrad_scheme: impl FaceNormalGradientScheme<DIM, Lhs = f64, Rhs = f64>,
    pressure_bc: impl Fn(&FaceRef<'a, DIM>) -> (f64, f64),
    mesh: &'a Mesh<DIM>,
) {
    for face in mesh.iter_faces() {
        let i = face.owner();

        let ainv = ainv_face[face.id()];
        let hbyan = hbyan_face[face.id()];

        let (pgti, pgtj, pgtrhs) = fngrad_scheme.terms(&face, mesh);

        match face.neighbor() {
            FaceNeighbor::Cell(j) => {

                // diffusion, central scheme

                let pgrad_n =
                    pgti * pressure[i]
                    + pgtj * pressure[j]
                    + pgtrhs
                ;

                phi[face.id()] = hbyan - ainv * pgrad_n;
            },
            FaceNeighbor::Boundary(_) => {
                let (blhsp, bvp) = pressure_bc(&face);

                let p_face = blhsp * pressure[i] + bvp;

                let pgrad_n =
                    pgti * pressure[i]
                    + pgtj * p_face
                    + pgtrhs
                ;

                phi[face.id()] = hbyan - ainv * pgrad_n;
            },
        }
    }

    // call update to collect phi from other ranks
    phi.update();

}



pub fn correct_velocity<const DIM: usize>(
    velocity: &mut Field<Vector<DIM>, geometry::Cell, DIM>,
    hbya: &Field<Vector<DIM>, geometry::Cell, DIM>,
    ainv: &Field<f64, geometry::Cell, DIM>,
    pressure_gradient: &Field<Vector<DIM>, geometry::Cell, DIM>,
    mesh: &Mesh<DIM>,
) {

    for cell in mesh.iter_cells() {
        let i = cell.id();

        let pgrad = pressure_gradient[i];

        // update velocity
        velocity[i] = hbya[i] - ainv[i] * pgrad;
    }
    velocity.update();

}





