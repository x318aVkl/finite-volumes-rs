

use std::collections::{HashMap, HashSet};

use crate::{Mesh, mesh::{CellIndex, FaceIndex, NodeIndex, Ownership}};




impl<const DIM: usize> Mesh<DIM> {

    pub fn decompose(&self, nparts: usize) -> Result<impl Iterator<Item = Result<Mesh<DIM>, crate::error::Error>>, crate::error::Error> {
        // can only be ran in serial
        assert!(self.mpi_comm.is_none());

        // compute the node wise decomposition
        let mut xadj: Vec<i32> = vec![0];
        let mut adjncy: Vec<i32> = vec![];

        for node in self.iter_nodes() {
            for other in self.node_to_node.major_range(usize::from(node.id())) {
                if *other != node.id() {
                    adjncy.push(usize::from(*other) as i32);
                }
            }

            xadj.push(adjncy.len() as i32);
        }

        let mut parts = vec![0; self.n_nodes()];

        metis::Graph::new(1, nparts as i32, &xadj, &adjncy)?.part_recursive(&mut parts)?;

        // renumber nodes according to their new partitions
        let mut old_to_new_node_numbering = vec![0; self.n_nodes()];
        let mut new_to_old_node_numbering = vec![0; self.n_nodes()];
        let mut max_node_id = 0;
        for part in 0..nparts {
            let part = part as i32;
            for i in 0..self.n_nodes() {
                if parts[i] == part {
                    old_to_new_node_numbering[i] = max_node_id;
                    new_to_old_node_numbering[max_node_id] = i;
                    max_node_id += 1;
                }
            }
        }


        // compute new face/cell ownerships
        let mut new_face_owners = vec![0i32; self.n_faces()];
        let mut new_cell_owners = vec![0i32; self.n_cells()];
        for face in self.iter_faces() {
            let mut nnodeown: HashMap<i32, usize> = HashMap::new();
            for node in face.nodes() {
                let owner = parts[usize::from(*node)];
                if nnodeown.contains_key(&owner) {
                    *nnodeown.get_mut(&owner).unwrap() += 1;
                } else {
                    nnodeown.insert(owner, 1);
                }
            }
            let mut nnodeown = nnodeown.iter().collect::<Vec<_>>();
            nnodeown.sort_by(|a, b| if a.1 == b.1 {a.0.cmp(&b.0)} else {a.1.cmp(&b.1)});
            let owner = nnodeown[0].0;
            new_face_owners[usize::from(face.id)] = *owner;
        }
        for cell in self.iter_cells() {
            let mut nnodeown: HashMap<i32, usize> = HashMap::new();
            for node in cell.nodes() {
                let owner = parts[usize::from(*node)];
                if nnodeown.contains_key(&owner) {
                    *nnodeown.get_mut(&owner).unwrap() += 1;
                } else {
                    nnodeown.insert(owner, 1);
                }
            }
            let mut nnodeown = nnodeown.iter().collect::<Vec<_>>();
            nnodeown.sort_by(|a, b| if a.1 == b.1 {a.0.cmp(&b.0)} else {a.1.cmp(&b.1)});
            let owner = nnodeown[0].0;
            new_cell_owners[usize::from(cell.id)] = *owner;
        }

        // partition
        let iterator = (0..nparts).map(move |part| {
            let part = part as i32;

            let mut partmesh = Mesh::new(None);

            // add the local nodes
            let mut local_node_max_idx = 0;
            let mut local_node_ids: Vec<Option<NodeIndex>> = vec![None; self.n_nodes()];
            for i in 0..self.n_nodes() {
                if parts[i] == part {
                    partmesh.add_node(self.node(NodeIndex::from(i)).position(), Ownership::Owned, Some(self.node_global_id[old_to_new_node_numbering[i]]));
                    local_node_ids[i] = Some(NodeIndex(local_node_max_idx));
                    local_node_max_idx += 1;
                }
            }

            // figure out the extra nodes needed, and add them in order
            let mut extra_nodes = HashSet::new();
            for cell in self.iter_cells() {
                let mut any_owned = false;
                for n in cell.nodes() {
                    if parts[usize::from(*n)] == part {
                        any_owned = true;
                        break;
                    }
                }
                if !any_owned {continue}

                // add extra nodes if needed
                for n in cell.nodes() {
                    if parts[usize::from(*n)] != part {
                        // we need this node from the other partition
                        extra_nodes.insert(old_to_new_node_numbering[usize::from(*n)]);
                    }
                }
            }
            let mut extra_nodes: Vec<_> = extra_nodes.into_iter().collect();
            extra_nodes.sort();
            for n in extra_nodes {
                let n = new_to_old_node_numbering[n];
                partmesh.add_node(self.node(NodeIndex::from(n)).position(), Ownership::Remote(parts[usize::from(n)] as usize), Some(self.node_global_id[old_to_new_node_numbering[n]]));
                local_node_ids[n] = Some(NodeIndex(local_node_max_idx));
                local_node_max_idx += 1;
            }
            

            let mut faces_to_add: Vec<(i32, usize)> = vec![];
            let mut cells_to_add: Vec<(i32, usize)> = vec![];

            let mut faces_added: HashSet<FaceIndex> = HashSet::new();
            
            for cell in self.iter_cells() {
                let mut any_owned = false;
                for n in cell.nodes() {
                    if parts[usize::from(*n)] == part {
                        any_owned = true;
                        break;
                    }
                }
                if !any_owned {continue}

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
                   if new_cell_owners[usize::from(cell.id)] == part {-1} else {new_cell_owners[usize::from(cell.id)]},
                    usize::from(cell.id),
                ));
            }


            // sort the faces and cells based on rank, then on id
            faces_to_add.sort_by(|a, b| if a.0 == b.0 {a.1.cmp(&b.1)} else {a.0.cmp(&b.0)});
            cells_to_add.sort_by(|a, b| if a.0 == b.0 {a.1.cmp(&b.1)} else {a.0.cmp(&b.0)});

            // add the faces
            let mut local_face_ids: Vec<Option<FaceIndex>> = vec![None; self.n_faces()];
            let mut max_face_id: usize = 0;

            for (rank, fid) in faces_to_add {
                // add the face
                let mut node_buffer = [NodeIndex(0); 64];
                let face = self.face(FaceIndex(fid));
                if face.n_nodes() > node_buffer.len() {
                    panic!("Face has too many nodes {} > {}", face.n_nodes(), node_buffer.len());
                }

                for (k, n) in face.nodes().iter().enumerate() {
                    let n = local_node_ids[usize::from(*n)].expect("partition mesh contains node");
                    node_buffer[k] = n;
                }

                partmesh.add_face(&node_buffer[0..face.n_nodes()], face.boundary(), if rank == -1 {Ownership::Owned} else {Ownership::Remote(rank as usize)}, Some(fid as u32));

                local_face_ids[usize::from(fid)] = Some(FaceIndex(max_face_id));
                max_face_id += 1;
            }

            // add the cells
            for (rank, cid) in cells_to_add {
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
                partmesh.add_cell(&face_buffer[0..cell.n_faces()],if rank == -1 {Ownership::Owned} else {Ownership::Remote(rank as usize)}, Some(cid as u32));
            }


            partmesh.compute()?;
            
            Ok(partmesh)
        });

        Ok(iterator)
    }
    

}

