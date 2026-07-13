

use std::collections::HashSet;

use crate::{Mesh, core::mesh::{CellIndex, FaceIndex, InternalFaceNeighbor, NodeIndex, Ownership}};




impl<const DIM: usize> Mesh<DIM> {

    pub fn decompose(&self, nparts: usize) -> Result<impl Iterator<Item = Result<Mesh<DIM>, crate::core::error::Error>>, crate::core::error::Error> {
        // can only be ran in serial
        assert!(self.mpi_comm.is_none());

        // compute the node wise decomposition
        let mut xadj: Vec<i32> = vec![0];
        let mut adjncy: Vec<i32> = vec![];

        for cell in self.iter_cells() {
            for other in self.cell_to_cell.major_range(usize::from(cell.id())) {
                if *other != cell.id() {
                    adjncy.push(usize::from(*other) as i32);
                }
            }

            xadj.push(adjncy.len() as i32);
        }

        let mut parts = vec![0; self.n_cells()];

        metis::Graph::new(1, nparts as i32, &xadj, &adjncy)?.part_recursive(&mut parts)?;

        // renumber cells according to their new partitions
        let mut old_to_new_cell_numbering = vec![0; self.n_cells()];
        let mut new_to_old_cell_numbering = vec![0; self.n_cells()];
        let mut max_cell_id = 0;
        for part in 0..nparts {
            let part = part as i32;
            for i in 0..self.n_cells() {
                if parts[i] == part {
                    old_to_new_cell_numbering[i] = max_cell_id;
                    new_to_old_cell_numbering[max_cell_id] = i;
                    max_cell_id += 1;
                }
            }
        }


        // compute new face ownerships
        // face is owned by the owned cells rank
        let mut new_face_owners = vec![0i32; self.n_faces()];
        let mut partnfaces = vec![0; nparts];
        for face in self.iter_faces() {
            let ocell = face.data.owner_cell;
            let ownerpart = parts[usize::from(ocell)];
            
            let fpart = match face.data.neighbor {
                InternalFaceNeighbor::Cell(c1) => {
                    let neighborpart = parts[usize::from(c1)];
                    if partnfaces[ownerpart as usize] <= partnfaces[neighborpart as usize] {
                        ownerpart
                    } else {
                        neighborpart
                    }
                },
                _ => {
                    ownerpart
                }
            };
            new_face_owners[usize::from(face.id())] = fpart;
            partnfaces[fpart as usize] += 1;
        }

        // partition
        let iterator = (0..nparts).map(move |part| {
            let part = part as i32;

            let mut partmesh = Mesh::new(None);

            // add the local nodes
            let mut local_node_max_idx = 0;
            let mut local_node_ids: Vec<Option<NodeIndex>> = vec![None; self.n_nodes()];

            let mut faces_to_add: Vec<(i32, usize)> = vec![];
            let mut cells_to_add: Vec<(i32, usize)> = vec![];

            let mut faces_added: HashSet<FaceIndex> = HashSet::new();
            
            for cell in self.iter_cells() {
                let owned = parts[usize::from(cell.id())] == part;

                let mut any_ncell_owned = false;
                if !owned {
                    // check if any face shares a cell that is owned,
                    // if so, we need it
                    for f in cell.faces() {
                        let face = self.face(*f);
                        let ocell = match face.other_cell(cell.id()) {
                            None => continue,
                            Some(v) => v,
                        };
                        let ocell_part = parts[usize::from(ocell)];
                        if ocell_part == part {
                            any_ncell_owned = true;
                            break;
                        }
                    }
                }
                let required = owned | any_ncell_owned;
                if !required {continue}

                // we need to add the faces
                for  f in cell.faces().iter() {
                    if !faces_added.contains(f) {

                        // add the face to faces added
                        let face = self.face(*f);

                        //partmesh.add_face(&node_buffer[0..face.n_nodes()], face.boundary());
                        faces_to_add.push((
                            if new_face_owners[usize::from(*f)] == part {-1} else {new_face_owners[usize::from(*f)]},
                            usize::from(face.id),
                        ));

                        faces_added.insert(*f);
                    };
                }

                //partmesh.add_cell(&face_buffer[0..cell.n_faces()]);
                cells_to_add.push((
                    if parts[usize::from(cell.id)] == part {-1} else {parts[usize::from(cell.id)]},
                    old_to_new_cell_numbering[usize::from(cell.id)],
                ));
            }


            // sort the faces and cells based on rank, then on id
            faces_to_add.sort_by(|a, b| if a.0 == b.0 {a.1.cmp(&b.1)} else {a.0.cmp(&b.0)});
            cells_to_add.sort_by(|a, b| if a.0 == b.0 {a.1.cmp(&b.1)} else {a.0.cmp(&b.0)});

            // add the faces
            let mut local_face_ids: Vec<Option<FaceIndex>> = vec![None; self.n_faces()];
            let mut max_face_id: usize = 0;

            let mut bndfacestart = self.n_faces();
            for (rank, fid) in faces_to_add {
                // add the face
                let mut node_buffer = [NodeIndex(0); 64];
                let face = self.face(FaceIndex(fid));
                if face.n_nodes() > node_buffer.len() {
                    panic!("Face has too many nodes {} > {}", face.n_nodes(), node_buffer.len());
                }

                for (k, n) in face.nodes().iter().enumerate() {
                    let n = match local_node_ids[usize::from(*n)] {
                        Some(n) => n,
                        None => {
                            // add it
                            partmesh.add_node(self.node(*n).position());
                            local_node_ids[usize::from(*n)] = Some(NodeIndex::from(local_node_max_idx));
                            local_node_max_idx += 1;
                            NodeIndex::from(local_node_max_idx - 1)
                        }
                    };
                    node_buffer[k] = n;
                }

                match face.boundary() {
                    Some(_) => {
                        bndfacestart = bndfacestart.min(max_face_id);
                    },
                    None => {}
                }

                partmesh.add_face(&node_buffer[0..face.n_nodes()], match face.boundary() {Some(b) => Some(b.0), None => None}, if rank == -1 {Ownership::Owned} else {Ownership::Remote(rank as usize)}, Some(fid as u32));

                local_face_ids[usize::from(fid)] = Some(FaceIndex(max_face_id));
                max_face_id += 1;
            }

            // add the cells
            for (rank, ncid) in cells_to_add {
                let cid = new_to_old_cell_numbering[ncid];
                // add the face
                let mut face_buffer = [FaceIndex(0); 64];
                let cell = self.cell(CellIndex(cid));
                if cell.n_faces() > face_buffer.len() {
                    panic!("Cell has too many faces {} > {}", cell.n_faces(), face_buffer.len());
                }

                for (k, f) in cell.faces().iter().enumerate() {
                    let f = local_face_ids[usize::from(*f)].expect("partition mesh contains face");
                    face_buffer[k] = f;
                }
                partmesh.add_cell(&face_buffer[0..cell.n_faces()],if rank == -1 {Ownership::Owned} else {Ownership::Remote(rank as usize)}, Some(ncid as u32));
            }

            // copy the patches
            for (name, id) in self.patch_name_ids.iter() {
                partmesh.add_patch(id.0, name.as_str(), Some((FaceIndex(bndfacestart), 0)))?;
            }

            partmesh.compute()?;
            
            Ok(partmesh)
        });

        Ok(iterator)
    }
    

}



#[cfg(test)]
mod test {

    #[test]
    fn decompose_square_mesh() {

        let mesh = super::super::examples::square().unwrap();

        // also try to decompose the mesh
        let mut total_volume = 0.0;
        for part in mesh.decompose(4).unwrap() {
            let part = part.unwrap();
            let mut subvolume = 0.0;
            for cell in part.iter_cells() {
                subvolume += cell.volume();
            }
            total_volume += subvolume;
        }
        assert!((total_volume - 1.0).abs() < f64::EPSILON*10.0);
        
    }

}


