use std::collections::HashSet;

use crate::{Matrix, Mesh, Sparsity, Vector, error::Error, mesh::{CellData, CellIndex, FaceData, FaceIndex, NodeIndex}, vector::Normal};





impl<const DIM: usize> Mesh<DIM> {


    pub fn compute(&mut self) -> Result<(), Error> {

        // compute the cell nodes
        {
            self.cell_nodes = Sparsity::new();
            for c in 0..self.n_total_cells() {
                let mut cnodes = HashSet::new();
                for f in self.cell_faces.major_range(c) {
                    for n in self.face_nodes.major_range(usize::from(*f)) {
                        cnodes.insert(*n);
                    }
                }
                for n in cnodes {
                    self.cell_nodes.push_to_major(n);
                }
                self.cell_nodes.close_major();
            }
            self.cell_nodes.sort();
        }

        // compute the node to node sparsity
        {
            let mut node_to_node = vec![HashSet::<NodeIndex>::new(); self.nodes.len()];
            for c in 0..self.n_total_cells() {
                for i in self.cell_nodes.major_range(c) {
                    for j in self.cell_nodes.major_range(c) {
                        node_to_node[usize::from(*i)].insert(*j);
                    }
                }
            }
            self.node_to_node = Sparsity::new();
            for i in 0..self.n_total_nodes() {
                for j in &node_to_node[i] {
                    self.node_to_node.push_to_major(*j);
                }
                self.node_to_node.close_major();
            }
            self.node_to_node.sort();
        }

        // compute the cell to cell sparsity
        // uses node neighbors
        {
            let mut node_cells = vec![HashSet::<CellIndex>::new(); self.n_total_nodes()];
            for cell in self.iter_all_cells() {
                for n in cell.nodes() {
                    node_cells[usize::from(*n)].insert(cell.id());
                }
            }
            for c in 0..self.n_total_cells() {
                let mut cell_other_cells= HashSet::<CellIndex>::new();
                for n in self.cell(CellIndex(c)).nodes() {
                    for oc in &node_cells[usize::from(*n)] {
                        cell_other_cells.insert(*oc);
                    }
                }
                for c in cell_other_cells {
                    self.cell_to_cell.push_to_major(c);
                }
                self.cell_to_cell.close_major();
            }
            self.cell_to_cell.sort();
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

        // compute the cell data
        {
            for c in 0..self.n_total_cells() {
                let size = self.cell_faces.major_range(c).len();

                let mut center = Vector::new();
                for n in self.cell_nodes.major_range(c) {
                    center += self.nodes[usize::from(*n)];
                }
                center /= size as f64;

                calc_cell_data(&self.face_data, self.cell_faces.major_range(c), center, &mut self.cell_data[c])?;
            }
        }

        // compute the cell gradient coefficients
        {
            self.cell_node_gradient_coefficients = vec![Vector::new(); self.cell_nodes.minor_len()];

            for c in 0..self.n_total_cells() {
                let mut g = Matrix::<DIM, DIM>::new();

                // (n - cc)^T * c = ai - a0
                // A * c = b
                // A^T * A * c = A^T * b
                // c = (A^T * A)^-1 A^T * b
                let n0 = self.cell_nodes.major_range(c)[0];
                let mut kdiag = None;
                for k in self.cell_nodes.major_start(c)..self.cell_nodes.major_end(c) {
                    let n = self.cell_nodes.flat_index(k);
                    if n == n0 {
                        kdiag = Some(k);
                        continue;
                    }

                    let dgc = self.nodes[usize::from(n)] - self.nodes[usize::from(n0)];

                    g += dgc.outer(dgc);
                }

                let ginv = g.inv()?;
                let kdiag = kdiag.unwrap();

                for k in self.cell_nodes.major_start(c)..self.cell_nodes.major_end(c) {
                    let n = self.cell_nodes.flat_index(k);
                    if n == n0 {continue}

                    let dgc = self.nodes[usize::from(n)] - self.nodes[usize::from(n0)];

                    let gi = ginv * dgc;

                    self.cell_node_gradient_coefficients[k] = gi;
                    self.cell_node_gradient_coefficients[kdiag] -= gi;

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



fn calc_cell_data<const DIM: usize>(faces: &[FaceData<DIM>], cell_faces: &[FaceIndex], node_ave: Vector<DIM>, cell_data: &mut CellData<DIM>) -> Result<(), Error> {
    if DIM == 1 {

        cell_data.volume = (faces[usize::from(cell_faces[0])].center - faces[usize::from(cell_faces[1])].center).norm();
        cell_data.center = node_ave;

        Ok(())
    } else if DIM == 2 {
        let mut center = Vector::new();
        let mut volume = 0.0;

        for f in cell_faces {
            let face = faces[usize::from(*f)];

            let dfc = face.center - node_ave;
            let height = dfc.dot(face.normal).abs();
            let volume_i = height * face.area / 2.0;
            let center_i = (face.center * 2.0 + node_ave) / 3.0;

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

            let dfc = face.center - node_ave;
            let height = dfc.dot(face.normal).abs();
            let volume_i = height * face.area / 3.0;
            let center_i = (face.center * 3.0 + node_ave) / 4.0;

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




