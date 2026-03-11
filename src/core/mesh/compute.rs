use std::collections::HashSet;

use crate::core::{Matrix, Mesh, Vector, error::Error, mesh::{CellData, CellIndex, FaceData, FaceIndex, FaceNeighbor, NodeIndex}, vector::Normal};





impl<const DIM: usize> Mesh<DIM> {


    pub fn compute(&mut self) -> Result<(), Error> {

        // validate that all faces have an owner cell
        for fi in 0..self.n_total_faces() {
            let fo = usize::from(self.face_data[fi].owner_cell);
            assert!(fo < self.n_total_cells());
        }

        // validate that all faces without face neighbors have non owned cells as owner
        for fi in 0..self.n_total_faces() {
            let n = self.face_data[fi].neighbor;
            match n {
                FaceNeighbor::None => {
                    // check that the owner cell is not owned
                    let owner_cell = usize::from(self.face_data[fi].owner_cell);
                    assert!(!self.cell_data[owner_cell].ownership.owned());
                },
                _ => {}
            }
        }

        // compute the face data
        {
            for f in 0..self.n_total_faces() {
                let size = self.face_nodes.major_range(f).len();

                let mut c = Vector::new();
                for n in self.face_nodes.major_range(f) {
                    c += self.nodes[usize::from(*n)];
                }
                c /= size as f64;

                calc_face_data(&self.nodes, self.face_nodes.major_range(usize::from(f)), c, &mut self.face_data[f])?;
            }
        }
        // normals might be in the wrong direction, will be corrected later

        // compute the cell data
        {
            for c in 0..self.n_total_cells() {
                let size = self.cell_faces.major_range(c).len();

                let mut center = Vector::new();
                for n in self.cell_faces.major_range(c) {
                    center += self.face_data[usize::from(*n)].center;
                }
                center /= size as f64;

                calc_cell_data(&self.face_data, self.cell_faces.major_range(c), center, &mut self.cell_data[c])?;
            }
        }

        // correct the face normal directions and ownership ids
        // always from owner cell to neighhbor cell
        // for faces between ranks, if face is owned, it goes from 
        {
             for f in 0..self.n_total_faces() {
                let owner = self.face_data[f].owner_cell;
                let owner = usize::from(owner);

                let inside_point = match self.face_data[f].neighbor {
                    FaceNeighbor::Cell(n) => {
                        let neighbor = usize::from(n);
                        if self.cell_data[owner].ownership.owned() && self.cell_data[neighbor].ownership.owned() {
                            // no need to flip this face
                            if !self.face_data[f].ownership.owned() {
                                panic!("internal face not owned");
                            }
                        } else {
                            if !self.face_data[f].ownership.owned() {

                            //}
                            //if !self.cell_data[owner].ownership.owned() {
                                // also flip the owner and neighbor
                                self.face_data[f].owner_cell = CellIndex::from(neighbor);
                                self.face_data[f].neighbor = FaceNeighbor::Cell(CellIndex::from(owner));
                            }
                        }
                        self.cell_data[usize::from(self.face_data[f].owner_cell)].center
                    },
                    _ => {
                        self.cell_data[owner].center
                    }
                };

                if self.face_data[f].normal.dot(self.face_data[f].center - inside_point) < 0.0 {
                    self.face_data[f].normal *= -1.0;
                }
            }
        }

        // compute the cell to cell sparsity
        // uses face neighbors
        {
            for c in 0..self.n_total_cells() {
                let mut cell_other_cells= HashSet::<CellIndex>::new();
                for f in self.cell_faces.major_range(c) {
                    match self.face(*f).other_cell(CellIndex::from(c)) {
                        Some(cu) => {cell_other_cells.insert(cu);},
                        _ => {},
                    }
                }
                self.cell_to_cell.push_to_major(CellIndex(c));
                for c in cell_other_cells {
                    self.cell_to_cell.push_to_major(c);
                }
                self.cell_to_cell.close_major();
            }
            // do not sort it
            //self.cell_to_cell.sort();
        }

        // compute the cell gradient coefficients
        {
            self.cell_face_gradient_coefficients = vec![Vector::new(); self.cell_faces.minor_len()];
            self.cell_diag_gradient_coefficients = vec![Vector::new(); self.n_total_cells()];

            for c in 0..self.n_cells() {
                let mut g = Matrix::<DIM, DIM>::new();

                let xc = self.cell_data[c].center;

                for k in self.cell_faces.major_start(c)..self.cell_faces.major_end(c) {
                    let f = self.cell_faces.flat_index(k);

                    let dgc = self.face_data[usize::from(f)].center - xc;

                    g += dgc.outer(dgc);
                }

                let ginv = g.inv()?;

                for k in self.cell_faces.major_start(c)..self.cell_faces.major_end(c) {
                    let f = self.cell_faces.flat_index(k);

                    let dgc = self.face_data[usize::from(f)].center - xc;

                    let gi = ginv * dgc;

                    self.cell_face_gradient_coefficients[k] = gi;
                    self.cell_diag_gradient_coefficients[c] -= gi;
                }
            }
        }

        self.computed = true;
        Ok(())
    }

}



fn calc_face_data<const DIM: usize>(all_nodes: &[Vector<DIM>], face_nodes: &[NodeIndex], node_ave: Vector<DIM>, face_data: &mut FaceData<DIM>) -> Result<(), Error> {
    if DIM == 1 {
        face_data.area = 1.0;
        face_data.center = all_nodes[usize::from(face_nodes[0])];
        face_data.normal = Vector::one();
        
        Ok(())
    } else if DIM == 2 {
        assert_eq!(face_nodes.len(), 2);

        let n0 = all_nodes[usize::from(face_nodes[0])];
        let n1 = all_nodes[usize::from(face_nodes[1])];

        let dn = n1 - n0;
        let dsn = dn.normal(dn);

        face_data.area = dsn.norm();
        face_data.normal = dsn / face_data.area;
        face_data.center = (n0 + n1) * 0.5;

        Ok(())
    } else if DIM == 3 {
        let mut center = Vector::new();
        let mut normal = Vector::new();
        let mut area = 0.0;

        for i in 0..face_nodes.len() {

            let n0 = all_nodes[usize::from(face_nodes[i])];
            let n1 = all_nodes[usize::from(face_nodes[if i == (face_nodes.len()-1) {0} else {i + 1}])];

            let dn0 = n0 - node_ave;
            let dn1 = n1 - node_ave;

            let dsn = dn0.normal(dn1);

            let area_i = dsn.norm();
            area += area_i;
            normal = dsn;
            center += (n0 + n1 + node_ave) * 1.0 / 3.0 * area_i;
        }
        assert!(area > std::f64::EPSILON);

        normal /= area;
        center /= area;

        face_data.area = area;
        face_data.center = center;
        face_data.normal = normal;

        Ok(())
    } else {
        Err(Error::InvalidDimension(DIM))
    }
}



fn calc_cell_data<const DIM: usize>(faces: &[FaceData<DIM>], cell_faces: &[FaceIndex], facec_ave: Vector<DIM>, cell_data: &mut CellData<DIM>) -> Result<(), Error> {
    if DIM == 1 {

        cell_data.volume = (faces[usize::from(cell_faces[0])].center - faces[usize::from(cell_faces[1])].center).norm();
        cell_data.center = facec_ave;

        Ok(())
    } else if DIM == 2 {
        let mut center = Vector::new();
        let mut volume = 0.0;

        for f in cell_faces {
            let face = faces[usize::from(*f)];

            let dfc = face.center - facec_ave;
            let height = dfc.dot(face.normal).abs();
            let volume_i = height * face.area / 2.0;
            let center_i = (face.center * 2.0 + facec_ave) / 3.0;

            center += center_i * volume_i;
            volume += volume_i;
        }

        cell_data.volume = volume;
        cell_data.center = center / volume;

        Ok(())
    } else if DIM == 3 {
        let mut center = Vector::new();
        let mut volume = 0.0;

        for f in cell_faces {
            let face = faces[usize::from(*f)];

            let dfc = face.center - facec_ave;
            let height = dfc.dot(face.normal).abs();
            let volume_i = height * face.area / 3.0;
            let center_i = (face.center * 3.0 + facec_ave) / 4.0;

            center += center_i * volume_i;
            volume += volume_i;
        }

        cell_data.volume = volume;
        cell_data.center = center / volume;

        Ok(())
    } else {
        Err(Error::InvalidDimension(DIM))
    }
}




