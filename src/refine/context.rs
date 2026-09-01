use std::{collections::{HashMap, HashSet}, io::Read};

use mpi::{topology::SimpleCommunicator, traits::{Communicator, CommunicatorCollectives, Destination, Source}};

use crate::{Mesh, Vector, core::{mesh::{FaceIndex, NodeIndex}, traits::Zero}};


pub struct RefinementContext<const DIM: usize> {
    grid: p4est::grid::Grid<u32>,
    mpi_comm: SimpleCommunicator,
}


pub fn initialize(world: &SimpleCommunicator) {
    p4est::env::initialize(world);
}



impl<const DIM: usize> RefinementContext<DIM> {
    pub fn read<T: Read>(source: T, mpi_comm: SimpleCommunicator) -> Result<Self, crate::error::Error> {
        assert!(DIM == p4est::consts::DIM);
        let grid = p4est::grid::Grid::from_su2(source, mpi_comm.duplicate())?;
        Ok(Self {
            grid,
            mpi_comm,
        })
    }

    pub fn partition(&mut self) {
        self.grid.partition();
    }

    pub fn refine_uniform(&mut self) {
        self.grid.refine_uniform();
    }

    pub fn mesh(&self) -> Result<Mesh<DIM>, crate::error::Error> {
        assert!(DIM == p4est::consts::DIM);

        let own_rank = self.mpi_comm.rank() as u32;

        let mut mesh = Mesh::new(Some(self.mpi_comm.duplicate()));

        // add the nodes in the right order
        let mut added_nodes = HashSet::<usize>::new();
        added_nodes.reserve(self.grid.local_len());

        let node_ids = p4est::grid::nodes::NodeNumbering::new(&self.grid);

        let mut nodes_to_add = vec![];
        nodes_to_add.reserve(self.grid.local_len());
        // self.grid.map_cells(|cell| {
        //     let ids = node_ids.cell_nodes(&cell);

        //     for i in 0..(2_usize.pow(DIM as u32)) {
        //         let c = cell.corner(i);
        //         let cid = ids[i];

        //         if !added_nodes.contains(&cid) {
        //             added_nodes.insert(cid);
        //             if nodes_to_add.len() <= cid {
        //                 nodes_to_add.resize(cid + 1, Vector::zero());
        //             }
        //             nodes_to_add[cid] = c.into();
        //         }
        //     }
        // });
        self.grid.map_faces(|face| {
            let ids = node_ids.face_nodes(&face);
            println!("{:?}", ids);
            for i in 0..(2_usize.pow((DIM-1) as u32)) {
                let c = face.corner(i);
                let cid = ids[i];
                if !added_nodes.contains(&cid) {
                    println!("rank {} add node {} {:?}", own_rank, cid, c);
                    added_nodes.insert(cid);
                    if nodes_to_add.len() <= cid {
                        nodes_to_add.resize(cid + 1, Vector::one() * 42.);
                    }
                    nodes_to_add[cid] = c.into();
                    println!("rank {} adding node {} {:?}", own_rank, cid, c);
                } else {
                    println!("rank {} saw again node {} {:?}", own_rank, cid, c);
                }
            }
        });

        // add the nodes
        println!("ndoes to add: {:?}", nodes_to_add);
        for node in nodes_to_add {
            mesh.add_node(node);
        }

        // now build the faces
        // we need the face global id information
        let mut n_local_faces = 0;
        let mut n_nonlocal_faces = 0;
        let mut n_total_faces = 0;
        self.grid.map_faces(|face| {

            let (_bnd, g1, r1) = if face.cell1.is_none() {
                (Some(0), false, own_rank)
            } else {
                let c1 = face.cell1.as_ref().unwrap();
                (None, c1.is_ghost, c1.owner_rank)
            };
            let owner_rank = if g1 && face.cell0.is_ghost {own_rank} else {
                let r0 = face.cell0.owner_rank;
                r0.min(r1)
            };
            let owned = own_rank == owner_rank;
            if owned {
                n_local_faces += 1;
            } else {
                n_nonlocal_faces += 1;
            }

            n_total_faces += 1;
        });
        let n_local_faces = n_local_faces;

        let mut all_ranks_nlocal_faces = vec![0; self.mpi_comm.size() as usize];
        {
            self.mpi_comm.any_process().all_gather_into(&n_local_faces, &mut all_ranks_nlocal_faces);
        }
        let mut all_rank_globalfaceid_starts = vec![0; all_ranks_nlocal_faces.len() + 1];
        for i in 0..all_ranks_nlocal_faces.len() {
            all_rank_globalfaceid_starts[i+1] = all_rank_globalfaceid_starts[i] + all_ranks_nlocal_faces[i];
        }
        // now we have the global face id range for all ranks
        let mut current_global_id = all_rank_globalfaceid_starts[own_rank as usize];

        // first add all the internal faces that are owned
        let mut cell_faces = vec![vec![]; self.grid.len_with_ghosts()];
        let mut cell_global_ids = vec![0; self.grid.len_with_ghosts()];
        let mut cell_owner_ranks = vec![None; self.grid.len_with_ghosts()];
        let mut faces_id_map = HashMap::<[usize; 3], usize>::new();

        let mut tosend_remote_faces = vec![];

        self.grid.map_faces(|face| {
            let fnodes = node_ids.face_nodes(&face);

            let n0 = mesh.node(NodeIndex::from(fnodes[0]));
            let n1 = mesh.node(NodeIndex::from(fnodes[1]));
            let dx = n1.position() - n0.position();
            if (dx[0].abs() > 1e-8) && (dx[1].abs() > 1e-8) {
                println!("error, diagonal face! {:?} {:?} {:?} {:?} {:?} {:?} {:?} {:?}", dx, face, face.corner(0), face.corner(1), n0.id(), n1.id(), n0.position(), n1.position());
            }

            //println!("{:?} {:?}", mesh.node(NodeIndex::from(fnodes[0])).position(), mesh.node(NodeIndex::from(fnodes[1])).position());
            
            let fc = (0..(2_usize.pow((DIM - 1) as u32))).map(|i| {
                NodeIndex::from(fnodes[i])
            }).collect::<Vec<_>>();

            let mut fh = vec![0; fc.len()];
            for i in 0..fh.len() {
                fh[i] = usize::from(fc[i]);
            }
            fh.sort();
            let mut face_hash: [usize; 3] = [0; 3];
            for i in 0..3.min(fc.len()) {
                face_hash[i] = fh[i];
            }

            let (bnd, g1, r1) = if face.cell1.is_none() {
                (Some(0), false, own_rank)
            } else {
                let c1 = face.cell1.as_ref().unwrap();
                (None, c1.is_ghost, c1.owner_rank)
            };
            let (owner_rank, ocgid) = if g1 && face.cell0.is_ghost {(face.cell0.owner_rank, face.cell0.global_id)} else {
                let r0 = face.cell0.owner_rank;
                //r0.min(r1)
                if r0 <= r1 {
                    (r0, face.cell0.global_id)
                } else {
                    (r1, face.cell1.as_ref().unwrap().global_id)
                }
            };
            let owned = own_rank == owner_rank;
                
            if bnd.is_none() && owned {

                let id = *faces_id_map.entry(face_hash).or_insert_with(|| {
                    println!("fc = {:?}, face nodes = {:?}", fc, fnodes);
                    mesh.add_face(&fc, bnd, crate::core::mesh::Ownership::Owned, Some(current_global_id));
                    current_global_id += 1;

                    if owned && (face.cell0.is_ghost || if let Some(c1) = &face.cell1 {c1.is_ghost} else {false}) {
                        // owned cell but either c0 or c1 is remote
                        // this faces global id will have to be sent
                        let other_rank = if face.cell0.owner_rank == owner_rank {face.cell1.as_ref().unwrap().owner_rank} else {face.cell0.owner_rank};
                        tosend_remote_faces.push((other_rank, ocgid, current_global_id - 1));
                    }

                    mesh.n_faces() - 1
                });

                cell_faces[face.cell0.local_id].push(FaceIndex::from(id));
                cell_global_ids[face.cell0.local_id] = face.cell0.global_id;
                cell_owner_ranks[face.cell0.local_id] = Some(face.cell0.owner_rank);
                if let Some(c1) = &face.cell1 {
                    cell_faces[c1.local_id].push(FaceIndex::from(id));
                    cell_global_ids[c1.local_id] = c1.global_id;
                    cell_owner_ranks[c1.local_id] = Some(c1.owner_rank);
                }
            }

        });


        // send the ordered remote faces global ids
        tosend_remote_faces.sort_by(|a, b| {
            if a.0 == b.0 {a.1.cmp(&b.1)} else {a.0.cmp(&b.0)}
        });


        let mut tosend_remote_faces_ids_only = vec![];
        for (_, _, id) in &tosend_remote_faces {
            tosend_remote_faces_ids_only.push(*id);
        }

        let mut remote_face_ids = HashMap::<u32, Vec<usize>>::new();
        let mut ordered_ranks = vec![];
        {
            let mut other_ranks = HashSet::new();
            for (i, _, _) in tosend_remote_faces.iter() {
                other_ranks.insert(*i);
            }

            let mut send_len_buffer = vec![0; other_ranks.len()];

            {
                let mut i = 0;
                for k in 0..send_len_buffer.len() {
                    let fs = i;
                    let mut fe = i + 1;
                    while (fe < tosend_remote_faces.len()) && (tosend_remote_faces[fe].0 == tosend_remote_faces[fs].0) {fe += 1;} 
                    ordered_ranks.push(tosend_remote_faces[fs].0);
                    i = fe;

                    let range_to_send = &tosend_remote_faces[fs..fe];
                    let send_len = range_to_send.len();
                    send_len_buffer[k] = send_len;
                    //self.mpi_comm.process_at_rank(curr_rank as i32).immediate_send(scope, &send_len_buffer[k]);
                }
            }
            let mut send_len_map = HashMap::new();
            for i in 0..send_len_buffer.len() {
                send_len_map.insert(ordered_ranks[i], send_len_buffer[i]);
            }

            let zero_val = 0;

            mpi::request::scope(|scope| {

                // tell other ranks how many shared faces we will send to them
                let reqs = (0..(self.mpi_comm.size() as u32)).map(|i| {
                    let send_len = if let Some(v) = send_len_map.get(&i) {v} else {&zero_val};
                    self.mpi_comm.process_at_rank(i as i32).immediate_send(scope, send_len)
                }).collect::<Vec<_>>();


                (0..(self.mpi_comm.size() as u32)).map(|orank| {

                    let (other_size, _) = self.mpi_comm.process_at_rank(orank as i32).receive::<usize>();
                    if other_size > 0 {
                        remote_face_ids.insert(orank, vec![0; other_size]);
                    }
                }).count();

                for r in reqs {
                    r.wait();
                }

                // send the actual values
                let mut i = 0;
                let reqs = (0..send_len_buffer.len()).map(|k| {
                    let fs = i;
                    let mut fe = i + 1;
                    while (fe < tosend_remote_faces.len()) && (tosend_remote_faces[fe].0 == tosend_remote_faces[fs].0) {fe += 1;} 
                    i = fe;

                    self.mpi_comm.process_at_rank(ordered_ranks[k] as i32).immediate_send(scope, &tosend_remote_faces_ids_only[fs..fe])
                }).collect::<Vec<_>>();

                for (orank, rfids) in remote_face_ids.iter_mut() {
                    let orank = *orank;

                    let _ = self.mpi_comm.process_at_rank(orank as i32).receive_into(rfids);
                }

                for r in reqs {
                    r.wait();
                }

            });
        }


        // now add all the boundary faces
        let mut bnd_faces_data = vec![];
        self.grid.map_faces(|face| {
            let fnodes = node_ids.face_nodes(&face);
            
            let fc = (0..(2_usize.pow((DIM - 1) as u32))).map(|i| {
                NodeIndex::from(fnodes[i])
            }).collect::<Vec<_>>();

            let (bnd, g1, r1) = if face.cell1.is_none() {
                (Some(0), false, own_rank)
            } else {
                let c1 = face.cell1.as_ref().unwrap();
                (None, c1.is_ghost, c1.owner_rank)
            };
            let (owner_rank, _ocgid) = if g1 && face.cell0.is_ghost {(own_rank, face.cell0.global_id)} else {
                let r0 = face.cell0.owner_rank;
                //r0.min(r1)
                if r0 <= r1 {
                    (r0, face.cell0.global_id)
                } else {
                    (r1, face.cell1.as_ref().unwrap().global_id)
                }
            };
            let owned = own_rank == owner_rank;

            if bnd.is_some() && owned {
                bnd_faces_data.push((bnd.unwrap(), fc, face.cell0.local_id));
                // let id = *faces_id_map.entry(face_hash).or_insert_with(|| {
                //     mesh.add_face(&fc, bnd, crate::core::mesh::Ownership::Owned, None);
                //     mesh.n_faces() - 1
                // });

                // cell_faces[face.cell0.local_id].push(FaceIndex::from(id));
                // cell_global_ids[face.cell0.local_id] = face.cell0.global_id;
                // if let Some(c1) = &face.cell1 {
                //     cell_faces[c1.local_id].push(FaceIndex::from(id));
                //     cell_global_ids[c1.local_id] = c1.global_id;
                // }
            }
        });
        bnd_faces_data.sort_by(|a, b| {a.0.cmp(&b.0)});
        for (bnd, fc, c0) in bnd_faces_data {
            let mut fh = vec![0; fc.len()];
            for i in 0..fh.len() {
                fh[i] = usize::from(fc[i]);
            }
            fh.sort();
            let mut face_hash: [usize; 3] = [0; 3];
            for i in 0..3.min(fc.len()) {
                face_hash[i] = fh[i];
            }
            let bnd = Some(bnd as u16);

            let id = *faces_id_map.entry(face_hash).or_insert_with(|| {
                println!("fc = {:?}", fc);
                mesh.add_face(&fc, bnd, crate::core::mesh::Ownership::Owned, Some(current_global_id));
                current_global_id += 1;
                mesh.n_faces() - 1
            });

            cell_faces[c0].push(FaceIndex::from(id));
        }

        // now add all the remote faces
        let mut remote_faces_data = vec![];
        self.grid.map_faces(|face| {
            
            let fnodes = node_ids.face_nodes(&face);
            
            let fc = (0..(2_usize.pow((DIM - 1) as u32))).map(|i| {
                NodeIndex::from(fnodes[i])
            }).collect::<Vec<_>>();

            let mut fh = vec![0; fc.len()];
            for i in 0..fh.len() {
                fh[i] = usize::from(fc[i]);
            }
            fh.sort();
            let mut face_hash: [usize; 3] = [0; 3];
            for i in 0..3.min(fc.len()) {
                face_hash[i] = fh[i];
            }

            let (bnd, g1, r1) = if face.cell1.is_none() {
                (Some(0), false, own_rank)
            } else {
                let c1 = face.cell1.as_ref().unwrap();
                (None, c1.is_ghost, c1.owner_rank)
            };
            let (owner_rank, ocgid) = if g1 && face.cell0.is_ghost {(own_rank, face.cell0.global_id)} else {
                let r0 = face.cell0.owner_rank;
                //r0.min(r1)
                if r0 <= r1 {
                    (r0, face.cell0.global_id)
                } else {
                    (r1, face.cell1.as_ref().unwrap().global_id)
                }
            };
            let owned = own_rank == owner_rank;

            if !owned {
                remote_faces_data.push((owner_rank, ocgid, fc, bnd, face.cell0.local_id, if let Some(c1) = &face.cell1 {Some(c1.local_id)} else {None}, face.cell0.global_id, if let Some(c1) = &face.cell1 {Some(c1.global_id)} else {None}, face.cell0.owner_rank, if let Some(c1) = &face.cell1 {Some(c1.owner_rank)} else {None}));
            }
                // let id = *faces_id_map.entry(face_hash).or_insert_with(|| {
                //     mesh.add_face(&fc, bnd, crate::core::mesh::Ownership::Remote(owner_rank as usize), None);
                //     mesh.n_faces() - 1
                // });

                // cell_faces[face.cell0.local_id].push(FaceIndex::from(id));
                // cell_global_ids[face.cell0.local_id] = face.cell0.global_id;
                // if let Some(c1) = &face.cell1 {
                //     cell_faces[c1.local_id].push(FaceIndex::from(id));
                //     cell_global_ids[c1.local_id] = c1.global_id;
                // }
        });

        // sort the remote faces by owner rank, and then by owner side cell global id (both rank will agree on this order, this is necessary)
        remote_faces_data.sort_by(|a, b| {
            if a.0 == b.0 {a.1.cmp(&b.1)} else {a.0.cmp(&b.0)}
        });

        println!("rmfdata = {:?}", remote_faces_data);

        let mut rank_offsets = HashMap::<u32, usize>::new();
        for (k, _, _, _, _, _, _, _, _, _) in &remote_faces_data {
            rank_offsets.insert(*k, 0);
        }
        
        for (owner_rank, _, fc, bnd, c0, c1, c0glob, c1glob, c0ownrank, c1ownrank) in remote_faces_data {
            let mut fh = vec![0; fc.len()];
            for i in 0..fh.len() {
                fh[i] = usize::from(fc[i]);
            }
            fh.sort();
            let mut face_hash: [usize; 3] = [0; 3];
            for i in 0..3.min(fc.len()) {
                face_hash[i] = fh[i];
            }

            let mut added = false;
            let id = *faces_id_map.entry(face_hash).or_insert_with(|| {
                // grab the global id for this face that was sent by the other rank
                let off = rank_offsets.get_mut(&owner_rank).unwrap();
                let gid = remote_face_ids.get(&owner_rank).unwrap()[*off];
                *off += 1;
                added = true;
                mesh.add_face(&fc, bnd, crate::core::mesh::Ownership::Remote(owner_rank as usize), Some(gid as u32));
                mesh.n_total_faces() - 1
            });
            if !added {
                panic!("error, a remote face was already added, this should not happen");
                // println!("rank {} setting face {} to remote {}", self.mpi_comm.rank(), id, owner_rank);
                // let off = rank_offsets.get_mut(&owner_rank).unwrap();
                // println!("{} {}", off, remote_face_ids.get(&owner_rank).unwrap().len());
                // println!("{} {:?}", self.mpi_comm.rank(), remote_face_ids);
                // let gid = remote_face_ids.get(&owner_rank).unwrap()[*off];
                // mesh.set_face_ownership(id, crate::core::mesh::Ownership::Remote(owner_rank as usize));
                // mesh.set_face_global_id(id, gid as u32);
                // *off += 1;
            }

            cell_faces[c0].push(FaceIndex::from(id));
            cell_global_ids[c0] = c0glob;
            cell_owner_ranks[c0] = Some(c0ownrank);
            if let Some(c1) = c1 {
                cell_faces[c1].push(FaceIndex::from(id));
                cell_global_ids[c1] = c1glob.unwrap();
                cell_owner_ranks[c1] = Some(c1ownrank.unwrap());
            }
        }


        // finally, add the cells
        let mut i = 0;
        for _ in 0..self.grid.local_len() {
            let faces = &mut cell_faces[i];
            if DIM == 2 {reorder_polygon_faces(faces.as_mut_slice(), &mesh);}
            let gid = cell_global_ids[i];
            mesh.add_cell(faces, crate::core::mesh::Ownership::Owned, Some(gid as u32));
            println!("rank {} added cell {:?}", own_rank, faces);
            i += 1;
        }
        for k in i..self.grid.len_with_ghosts() {
            if cell_faces[k].len() > 0 {
                //println!("rank {} adding ghost cell: {:?} {:?} {:?}", own_rank, &cell_faces[k], cell_global_ids[k], cell_owner_ranks[k]);
                let faces = &cell_faces[k];
                let gid = cell_global_ids[k];
                let owner_rank = cell_owner_ranks[k].unwrap();
                mesh.add_cell(faces, crate::core::mesh::Ownership::Remote(owner_rank as usize), Some(gid as u32));
            }
        }

        // add patches
        mesh.add_patch(0, "default", None)?;

        mesh.compute()?;

        Ok(unsafe { Mesh::<DIM>::clone(&*(&mesh as *const Mesh<{p4est::consts::DIM}> as *const Mesh<DIM>)) })
    }
}



fn reorder_polygon_faces<const DIM: usize>(
    faces: &mut [FaceIndex],
    mesh: &Mesh<DIM>,
) {
    let tol = 1e-10;

    let mut result = vec![FaceIndex::from(0); faces.len()];
    result[0] = faces[0];

    let mut to_add = faces.iter().map(|f| *f).collect::<Vec<_>>();
    to_add.remove(0);

    for i in 1..faces.len() {
        let n0 = mesh.node(mesh.face(result[i-1]).nodes()[0]).position();
        let n1 = mesh.node(mesh.face(result[i-1]).nodes()[1]).position();
        
        let mut found = None;
        for j in 0..to_add.len() {
            let p0 = mesh.node(mesh.face(to_add[j]).nodes()[0]).position();
            let p1 = mesh.node(mesh.face(to_add[j]).nodes()[1]).position();

            if ((p0 - n0).norm() < tol) || ((p1 - n1).norm() < tol) || ((p0 - n1).norm() < tol) || ((p1 - n0).norm() < tol) {
                found = Some(j);
                break;
            }
        }

        let found = found.unwrap();

        let fid = to_add.remove(found);
        result[i] = fid;
    }

    for i in 0..faces.len() {
        faces[i] = result[i];
    }
}