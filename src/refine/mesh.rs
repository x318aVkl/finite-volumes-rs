use std::collections::{HashMap, HashSet};

use crate::{Mesh, Vector, core::{Sparsity, mesh::{NodeIndex, Ownership, PatchIndex}}, prelude::{FaceIndex, FaceNeighbor, Zero}, refine::context::RefCommand};

// note: does not work for one dimensional meshes
// only 2D or 3D meshes


#[derive(Clone, Copy, Debug)]
pub struct EdgeData {
    child_edges: Option<(usize, usize)>,
    parent_edge: Option<usize>,
    child_middle_node: Option<usize>,
    refinement_level: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct FaceData {
    boundary: Option<PatchIndex>,
    owner: Ownership,
    refined_fstart: Option<usize>,
    refined_size: Option<usize>,
    refined_centernode: Option<usize>,
    parent_face: Option<usize>,
    refinement_level: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct CellData {
    owner: Ownership,
    original_cell: Option<usize>,
    parent_cell: Option<usize>,
    refined_cstart: Option<usize>,
    refined_len: Option<usize>,
    refinement_level: usize,
}


pub struct RefinementMesh<const DIM: usize> {

    nodes: Vec<Vector<DIM>>,

    edge_nodes: Vec<(usize, usize)>,

    face_edges: Sparsity<usize>,

    cell_faces: Sparsity<usize>,

    edge_data: Vec<EdgeData>,

    face_data: Vec<FaceData>,

    cell_data: Vec<CellData>,


    edge_hash: HashMap<(usize, usize), usize>,


    patch_info: Vec<(String, PatchIndex)>,

    cell_leaf_ids: Vec<Option<usize>>,
    cell_leaf_to_local_id: HashMap<usize, usize>,

    leaf_node_neighbors: Sparsity<usize>,

}



impl<const DIM: usize> RefinementMesh<DIM> {

    pub fn from_mesh(mesh: &Mesh<DIM>) -> Self {

        let mut rmesh = RefinementMesh {
            nodes: vec![],
            edge_nodes: vec![],
            face_edges: Sparsity::new(),
            cell_faces: Sparsity::new(),
            edge_data: vec![],
            face_data: vec![],
            cell_data: vec![],
            edge_hash: HashMap::new(),
            patch_info: vec![],
            cell_leaf_ids: vec![],
            cell_leaf_to_local_id: HashMap::new(),
            leaf_node_neighbors: Sparsity::new(),
        };

        for patch in mesh.iter_patch() {
            rmesh.patch_info.push((patch.name().to_string(), patch.id()));
        }

        for n in mesh.iter_nodes() {
            rmesh.nodes.push(n.center());
        }

        let mut edges_hash = HashMap::<(usize, usize), usize>::new();
        let mut node_cells = HashMap::<usize, HashSet<usize>>::new();

        for f in mesh.iter_all_faces() {

            for i in 0..f.nodes().len() {

                let j = if i == (f.nodes().len() - 1) {0} else {i+1};
                if (DIM == 2) && (j == 0) {break;}

                let i = usize::from(f.nodes()[i]);
                let j = usize::from(f.nodes()[j]);

                let edge = (i.min(j), i.max(j));

                let edge = match edges_hash.get(&edge) {
                    Some(v) => *v,
                    None => {
                        edges_hash.insert(edge, rmesh.edge_nodes.len());
                        rmesh.edge_nodes.push(edge);
                        rmesh.edge_nodes.len() - 1
                    }
                };

                let c0 = usize::from(f.owner());
                match f.neighbor() {
                    FaceNeighbor::Cell(c1) => {
                        let c1 = usize::from(c1);

                        match node_cells.get_mut(&i) {
                            Some(nc) => {
                                nc.insert(c0);
                                nc.insert(c1);
                            },
                            None => {
                                let mut nc = HashSet::new();
                                nc.insert(c0);
                                nc.insert(c1);
                                node_cells.insert(i, nc);
                            }
                        }
                    },
                    FaceNeighbor::Boundary(_) => {
                        match node_cells.get_mut(&i) {
                            Some(nc) => {
                                nc.insert(c0);
                            },
                            None => {
                                let mut nc = HashSet::new();
                                nc.insert(c0);
                                node_cells.insert(i, nc);
                            }
                        }
                    }
                };

                rmesh.face_edges.push_to_major(edge);
            }
            rmesh.face_edges.close_major();
            rmesh.face_data.push(FaceData { boundary: f.boundary(), owner: f.ownership(), refined_fstart: None, refined_size: None, refined_centernode: None, parent_face: None, refinement_level: 0, });
        }

        rmesh.edge_data = vec![EdgeData {child_edges: None, parent_edge: None, child_middle_node: None, refinement_level: 0}; edges_hash.len()];
        rmesh.edge_hash = edges_hash;

        for c in mesh.iter_all_cells() {
            for f in c.faces() {
                let f = usize::from(*f);
                rmesh.cell_faces.push_to_major(f);
            }
            rmesh.cell_faces.close_major();
            rmesh.cell_data.push(CellData { owner: c.ownership(), original_cell: Some(usize::from(c.id())), parent_cell: None, refined_cstart: None, refined_len: None, refinement_level: 0 });
            let cid = usize::from(c.id());
            rmesh.cell_leaf_ids.push(Some(cid));
            rmesh.cell_leaf_to_local_id.insert(cid, cid);

            let mut cell_to_cell = HashSet::new();
            for f in c.faces() {
                let f = mesh.face(*f);
                for n in f.nodes() {
                    let n = usize::from(*n);
                    for oc in node_cells.get(&n).unwrap() {
                        cell_to_cell.insert(*oc);
                    }
                }
            }
            for oc in cell_to_cell {
                if oc != usize::from(c.id()) {
                    rmesh.leaf_node_neighbors.push_to_major(oc);
                }
            }
            rmesh.leaf_node_neighbors.close_major();
        }

        rmesh
    }



    pub fn compute_refinement_order(&self, order: &mut Vec<Option<(usize, super::context::RefCommand)>>, criteria: &[f64], level: f64) {
        order.resize(self.cell_data.len(), None);

        // first figure out which cells have to be refined due do the criteria
        // without balancing
        for c in 0..self.cell_data.len() {
            
            // do not refine again cells that are already refined, so that dont have an original cell
            // only refine leaf cells
            // let orig_cell = if let Some(co) = self.cell_data[c].original_cell {
            //     co
            // } else {
            //     continue;
            // };
            let leaf_id = if let Some(co) = self.cell_leaf_ids[c] {
                co
            } else {
                continue;
            };
            
            let crit = criteria[leaf_id];

            if crit > level {
                // refinement needed
                order[c] = Some((0, RefCommand::Refine));
            }
        }

        // now do the rebalancing process
        // any cell that has neighbor refined cells that any connected leaf will be refined
        // also has to be refined
        loop {
            let mut nupdated = 0;
            for c in 0..self.cell_data.len() {

                // no need to check cells that are already refined
                if order[c].is_some() {
                    continue;
                }

                // only check leaf cells
                let leaf_id = if let Some(co) = self.cell_leaf_ids[c] {
                    co
                } else {
                    continue;
                };

                let orig_ref_level = self.cell_data[c].refinement_level;

                for leaf_neighbor in self.leaf_node_neighbors.major_range(leaf_id) {
                    let local_neighbor = *self.cell_leaf_to_local_id.get(leaf_neighbor).unwrap();

                    let leaf_ref_level = self.cell_data[local_neighbor].refinement_level;

                    if (leaf_ref_level > orig_ref_level) && (order[local_neighbor].is_some()) {
                        // we need to refine this cell before the leaf neighbor
                        let order_n = order[local_neighbor].unwrap();
                        order[c] = Some((order_n.0 + 1, RefCommand::Refine));
                        nupdated += 1;
                        break;
                    }
                }

            }
            //println!("nupdated = {}", nupdated);
            if nupdated == 0 {
                break;
            }
        }
    }



    pub fn refine(&mut self, order: &[Option<(usize, RefCommand)>]) {

        // figure out the max order
        let mut max_order = 0;
        for i in 0..order.len() {
            if let Some(oi) = order[i] {
                max_order = max_order.max(oi.0);
            }
        }

        for current_order in (0..=max_order).rev() {

            let mut c = 0;
            let mut started: bool = false;
            loop {
                if started {
                    c += 1;
                }
                started = true;
                if c >= self.cell_data.len() {
                    break;
                }
                if c >= self.cell_leaf_ids.len() {
                    // we went over all the previous cells and are in the new leaf cells
                    // do not refine them
                    break;
                }

                
                
                // do not refine again cells that are already refined, so that dont have an original cell
                // only refine leaf cells
                // let orig_cell = if let Some(co) = self.cell_data[c].original_cell {
                //     co
                // } else {
                //     continue;
                // };
                // let leaf_id = if let Some(co) = self.cell_leaf_ids[c] {
                //     co
                // } else {
                //     continue;
                // };

                // only run refinement for leaf cells
                if self.cell_leaf_ids[c].is_none() {
                    continue;
                }
                
                if let Some((order, _command)) = order[c] {

                    if order == current_order {
                        // refinement needed

                        self.refine_cell(c);
                    }
                }
            }
        }

        // update the local cell ids
        // and the leaf cell to cell node connectivity
        let mut leaf_cell_id: usize = 0;
        self.cell_leaf_ids.clear();
        self.cell_leaf_ids.resize(self.cell_data.len(), None);
        self.cell_leaf_to_local_id.clear();
        for c in 0..self.cell_data.len() {
            // if this cell is refined, skip it
            if self.cell_data[c].refined_cstart.is_some() {
                continue;
            }

            // leaf cell, add its id
            self.cell_leaf_ids[c] = Some(leaf_cell_id);
            self.cell_leaf_to_local_id.insert(leaf_cell_id, c);
            leaf_cell_id += 1;
        }

        // rebuild the cell to cell connectivity
        let nleafcells = leaf_cell_id;

        let mut node_leaf_cells = HashMap::<usize, HashSet<usize>>::new();
        node_leaf_cells.reserve(self.nodes.len());

        let mut leaf_cell_nodes = HashMap::<usize, HashSet<usize>>::new();
        leaf_cell_nodes.reserve(nleafcells);

        for c in 0..self.cell_data.len() {

            // only add it to the map if its a leaf node
            let leaf_id = if let Some(co) = self.cell_leaf_ids[c] {
                co
            } else {
                continue;
            };

            for f in self.cell_faces.major_range(c) {
                if self.face_data[*f].refined_fstart.is_some() {
                    // this is a refined face
                    let rfstart = self.face_data[*f].refined_fstart.unwrap();
                    let rfsize = self.face_data[*f].refined_size.unwrap();
                    for subface in rfstart..(rfstart + rfsize) {
                        assert!(self.face_data[subface].refined_fstart.is_none());

                        for e in self.face_edges.major_range(subface) {
                            let (n0, n1) = self.edge_nodes[*e];

                            for n in [n0, n1] {
                                match node_leaf_cells.get_mut(&n) {
                                    Some(nlfc) => {
                                        nlfc.insert(leaf_id);
                                    },
                                    None => {
                                        let mut nlfc = HashSet::new();
                                        nlfc.insert(leaf_id);
                                        node_leaf_cells.insert(n, nlfc);
                                    }
                                }
                                match leaf_cell_nodes.get_mut(&leaf_id) {
                                    Some(nlfc) => {
                                        nlfc.insert(n);
                                    },
                                    None => {
                                        let mut nlfc = HashSet::new();
                                        nlfc.insert(n);
                                        leaf_cell_nodes.insert(leaf_id, nlfc);
                                    }
                                }
                            }
                        }
                    }
                } else {
                    // this face is not refined
                    for e in self.face_edges.major_range(*f) {
                        let (n0, n1) = self.edge_nodes[*e];

                        for n in [n0, n1] {
                            match node_leaf_cells.get_mut(&n) {
                                Some(nlfc) => {
                                    nlfc.insert(leaf_id);
                                },
                                None => {
                                    let mut nlfc = HashSet::new();
                                    nlfc.insert(leaf_id);
                                    node_leaf_cells.insert(n, nlfc);
                                }
                            }
                            match leaf_cell_nodes.get_mut(&leaf_id) {
                                Some(nlfc) => {
                                    nlfc.insert(n);
                                },
                                None => {
                                    let mut nlfc = HashSet::new();
                                    nlfc.insert(n);
                                    leaf_cell_nodes.insert(leaf_id, nlfc);
                                }
                            }
                        }
                    }
                }
            }
        }


        // now that we have the leaf cell connectivity, update it to its compact sparsity form
        self.leaf_node_neighbors = Sparsity::new();
        for leaf_id in 0..nleafcells {

            for node in leaf_cell_nodes.get(&leaf_id).unwrap() {
                for ocell in node_leaf_cells.get(node).unwrap() {
                    if *ocell != leaf_id {
                        self.leaf_node_neighbors.push_to_major(*ocell);
                    }
                }
            }

            self.leaf_node_neighbors.close_major();
        }

    }



    fn refine_cell(&mut self, cell: usize) {

        let orig_cell_data = self.cell_data[cell];
        let orig_refinement_level = orig_cell_data.refinement_level;
        
        // we need the unique nodes to build the cells afterwards
        let mut nodes = HashSet::new();

        // we also need the node faces and edges
        let mut node_edges = HashMap::<usize, HashSet<usize>>::new();
        let mut node_faces = HashMap::<usize, HashSet<usize>>::new();

        for k in self.cell_faces.major_start(cell)..self.cell_faces.major_end(cell) {
            let f = self.cell_faces.flat_index(k);

            for edge in self.face_edges.major_range(f) {
                let (n0, n1) = self.edge_nodes[*edge];
                nodes.insert(n0);
                nodes.insert(n1);

                for n in [n0, n1] {
                    if node_faces.contains_key(&n) {
                        node_faces.get_mut(&n).unwrap().insert(f);
                    } else {
                        let mut nfset = HashSet::new();
                        nfset.insert(f);
                        node_faces.insert(n, nfset); 
                    }

                    if node_edges.contains_key(&n) {
                        node_edges.get_mut(&n).unwrap().insert(*edge);
                    } else {
                        let mut neset = HashSet::new();
                        neset.insert(*edge);
                        node_edges.insert(n, neset); 
                    }
                }
            }

            if self.face_data[f].refined_fstart.is_none() {
                // this face has not already been refined, we can/have to refine it
                self.refine_face(f);
            }
        }

        // build the new faces connecting the cell center to edges
        // we need the edge faces
        let mut edge_faces = HashMap::<usize, (usize, Option<usize>)>::new();
        for k in self.cell_faces.major_start(cell)..self.cell_faces.major_end(cell) {
            let f = self.cell_faces.flat_index(k);

            for e in self.face_edges.major_range(f) {
                let e = *e;

                match edge_faces.get_mut(&e) {
                    Some(ef) => {
                        assert!(ef.1.is_none());
                        ef.1 = Some(f);
                    },
                    None => {
                        edge_faces.insert(e, (f, None));
                    }
                }
            }
        }

        // create the refined cells
        // one per unique node

        // add the node average as center
        let center_node = self.nodes.len();
        let mut cn = Vector::zero();
        for i in &nodes {
            cn += self.nodes[*i];
        }
        self.nodes.push(cn / (nodes.len() as f64));


        // now make the new faces
        let mut edge_newfaces = HashMap::<usize, usize>::new();
        for (edge, faces) in edge_faces.iter() {
            let edge = *edge;

            let ecn = self.edge_data[edge].child_middle_node.unwrap();

            let f0 = faces.0;
            let f1 = faces.1.unwrap();

            let f0cn = self.face_data[f0].refined_centernode.unwrap();
            let f1cn = self.face_data[f1].refined_centernode.unwrap();

            let efe0 = self.find_edge((ecn, f0cn)).unwrap();
            let efe1 = self.find_edge((ecn, f1cn)).unwrap();
            

            // add the face center to cell center edge
            let ec0 = self.add_edge((f0cn, center_node), EdgeData { child_edges: None, parent_edge: None, child_middle_node: None, refinement_level: orig_refinement_level + 1 });
            let ec1 = self.add_edge((f1cn, center_node), EdgeData { child_edges: None, parent_edge: None, child_middle_node: None, refinement_level: orig_refinement_level + 1 });

            // add the face
            self.face_edges.push_to_major(efe0);
            self.face_edges.push_to_major(ec0);
            self.face_edges.push_to_major(ec1);
            self.face_edges.push_to_major(efe1);
            self.face_edges.close_major();

            self.face_data.push(FaceData { boundary: None, owner: orig_cell_data.owner, refined_fstart: None, refined_size: None, refined_centernode: None, parent_face: None, refinement_level: orig_refinement_level + 1 });

            edge_newfaces.insert(edge, self.face_data.len() - 1);
        }


        // one new cell per cell nodes
        let cstart = self.cell_data.len();
        let rlen = nodes.len();
        for node in nodes {

            let mut newfaces = vec![];

            // we need the edge newfaces corresponding to thisnodes edges
            for edge in node_edges.get(&node).unwrap() {
                let eface = edge_newfaces.get(edge).unwrap();

                newfaces.push(*eface);
            }

            // one face per subcell 
            // we need the subface corresponding to that node in each node faces
            for rootface in node_faces.get(&node).unwrap() {
                let subfaces_start = self.face_data[*rootface].refined_fstart.unwrap();
                let nsubface = self.face_data[*rootface].refined_size.unwrap();
                
                let mut subface = None;
                for f in subfaces_start..(subfaces_start+nsubface) {
                    for e in self.face_edges.major_range(f) {
                        let (en0, en1) = self.edge_nodes[*e];

                        if (en0 == node) || (en1 == node) {
                            subface = Some(f);
                            break;
                        }
                    }
                    if subface.is_some() {
                        break;
                    }
                }
                let subface = subface.unwrap();

                newfaces.push(subface);
            }

            // we have everything, now create the cell
            for f in newfaces {
                self.cell_faces.push_to_major(f);
            }
            self.cell_faces.close_major();

            self.cell_data.push(CellData { owner: orig_cell_data.owner, original_cell: None, parent_cell: Some(cell), refined_cstart: None, refined_len: None, refinement_level: orig_refinement_level + 1 })

        }

        // adjust the cell data
        self.cell_data[cell].refined_cstart = Some(cstart);
        self.cell_data[cell].refined_len = Some(rlen);
    }


    fn refine_face(&mut self, face: usize) {

        let owner_data = self.face_data[face];
        let orig_refinement_level = owner_data.refinement_level;

        let mut nodes = HashSet::new();
        let mut node_edges = HashMap::<usize, HashSet<usize>>::new();

        for k in self.face_edges.major_start(face)..self.face_edges.major_end(face) {
            let e = self.face_edges.flat_index(k);
            let (n0, n1) = self.edge_nodes[e];
            nodes.insert(n0);
            nodes.insert(n1);

            for n in [n0, n1] {
                match node_edges.get_mut(&n) {
                    Some(ne) => {ne.insert(e);},
                    None => {
                        let mut ne = HashSet::new();
                        ne.insert(e);
                        node_edges.insert(n, ne);
                    }
                }
            }

            // we have to refine this edge

            if self.edge_data[e].child_edges.is_none() {
                // refine the edge
                self.refine_edge(e);
            }
        }

        let size = nodes.len();

        // get and add the mid node
        let center_node = self.nodes.len();
        let mut cn = Vector::zero();
        for i in &nodes {
            cn += self.nodes[*i];
        }
        self.nodes.push(cn / (size as f64));

        // build the new edges connecting the cell center to edges
        let mut new_edges_centeredge = HashMap::<usize, usize>::new();
        for k in self.face_edges.major_start(face)..self.face_edges.major_end(face) {
            let e = self.face_edges.flat_index(k);

            let nmid = self.edge_data[e].child_middle_node.unwrap();

            let newedge = self.add_edge((nmid, center_node), EdgeData { child_edges: None, parent_edge: None, child_middle_node: None, refinement_level: orig_refinement_level + 1 });
            
            new_edges_centeredge.insert(e, newedge);
        }

        // build the refined subfaces
        let refined_fstart = self.face_data.len();
        let refined_size = nodes.len();
        for node in nodes {

            // make a face from (nfi-1, nfi), nfi, (nfi, nfi+1), fc

            // meaning, newedge from (nfi-1, fc) and (nfi+1, fc)
            // only if this edge does not already exists

            let mut newedges = vec![];

            for edge in node_edges.get(&node).unwrap() {
                let edge = *edge;
                // this edge was refined
                // get this nodes center edge

                let (e0, e1) = self.edge_data[edge].child_edges.unwrap();
                
                let eref = {
                    if (self.edge_nodes[e0].0 == node) || (self.edge_nodes[e0].1 == node) {
                        e0
                    } else {
                        e1
                    }
                };
                if !((self.edge_nodes[eref].0 == node) || (self.edge_nodes[eref].1 == node)) {
                    println!("ERROR for node {}", node);
                    println!("{:?}", self.edge_nodes[edge]);
                    println!("{:?}", self.edge_nodes[e0]);
                    println!("{:?}", self.edge_nodes[e1]);
                    panic!();
                }

                let ecent = *new_edges_centeredge.get(&edge).unwrap();

                if newedges.len() == 0 {
                    newedges.push(eref);
                    newedges.push(ecent);
                } else {
                    newedges.push(ecent);
                    newedges.push(eref);
                }
            }

            assert_eq!(newedges.len(), 4);

            // build the new refined face
            for e in newedges {
                self.face_edges.push_to_major(e);
            }
            self.face_edges.close_major();

            self.face_data.push(FaceData { boundary: owner_data.boundary, owner: owner_data.owner, refined_fstart: None, refined_size: None, refined_centernode: None, parent_face: Some(face), refinement_level: orig_refinement_level + 1 });
        }

        // update the face data
        self.face_data[face].refined_fstart = Some(refined_fstart);
        self.face_data[face].refined_size = Some(refined_size);
        self.face_data[face].refined_centernode = Some(center_node);

        // done!
    }

    fn refine_edge(&mut self, edge: usize) {

        // guard against refinining already refined edges
        if self.edge_data[edge].child_edges.is_some() {return}
        let orig_refinement_level = self.edge_data[edge].refinement_level;

        let (n0, n1) = self.edge_nodes[edge];

        let new_node = (self.nodes[n0] + self.nodes[n1]) * 0.5;
        self.nodes.push(new_node);
        let n2 = self.nodes.len() - 1;

        // add the new edges

        let ne0 = self.add_edge((n0, n2), EdgeData { child_edges: None, parent_edge: Some(edge), child_middle_node: None, refinement_level: orig_refinement_level + 1 });
        let ne1 = self.add_edge((n1, n2), EdgeData { child_edges: None, parent_edge: Some(edge), child_middle_node: None, refinement_level: orig_refinement_level + 1 });

        // update this edge data
        self.edge_data[edge].child_edges = Some((ne0, ne1));
        self.edge_data[edge].child_middle_node = Some(n2);

        // done!
    }


    fn add_edge(&mut self, edge: (usize, usize), data: EdgeData) -> usize {
        let edge = (edge.0.min(edge.1), edge.0.max(edge.1));

        // try to add it
        // if it already exists, return the existing value
        let edge_id = match self.edge_hash.get(&edge) {
            Some(id) => {
                return *id;
            },
            None => {
                self.edge_hash.insert(edge, self.edge_nodes.len());
                self.edge_nodes.len()
            }
        };

        self.edge_nodes.push(edge);
        self.edge_data.push(data);

        edge_id
    }


    fn find_edge(&self, edge: (usize, usize)) -> Option<usize> {
        let edge = (edge.0.min(edge.1), edge.0.max(edge.1));
        self.edge_hash.get(&edge).copied()
    }

}





impl<const DIM: usize> RefinementMesh<DIM> {

    pub fn build_mesh(&self) -> Mesh<DIM> {
        // serial for now!
        let mut mesh = Mesh::<DIM>::new(None);

        for patch in &self.patch_info {
            mesh.add_patch( u16::from(patch.1), patch.0.as_str(), None).unwrap();
        }

        for n in &self.nodes {
            mesh.add_node(*n);
        }

        let mut old_to_new_face_id: Vec<Option<usize>> = vec![None; self.face_data.len()];

        // first, add all the internal owned faces
        for f in 0..self.face_data.len() {
            if self.face_data[f].refined_centernode.is_some() {
                // this is a refined parent face, do not use
                continue;
            }

            if !self.face_data[f].owner.owned() {
                continue;
            }
            if self.face_data[f].boundary.is_some() {
                continue;
            }

            // this face is not a refined parent face
            // we can add it

            // get its nodes in order
            let mut nodes: Vec<NodeIndex> = vec![];
            //let nodes_e1 = self.edge_nodes[self.face_edges.major_range(f)[1]];
            let nodes_e1 = self.edge_nodes[self.face_edges.major_range(f)[self.face_edges.major_range(f).len() - 1]];
            for e in self.face_edges.major_range(f) {
                let (n0, n1) = self.edge_nodes[*e];

                if nodes.len() == 0 {
                    // add only the first one, the node shared by the end edge
                    if (n0 == nodes_e1.0) || (n0 == nodes_e1.1) {
                        nodes.push(NodeIndex::from(n0));
                        if let Some(n2) = self.edge_data[*e].child_middle_node {
                            nodes.push(NodeIndex::from(n2));
                        }
                        nodes.push(NodeIndex::from(n1));
                    } else if (n1 == nodes_e1.0) || (n1 == nodes_e1.1) {
                        nodes.push(NodeIndex::from(n1));
                        if let Some(n2) = self.edge_data[*e].child_middle_node {
                            nodes.push(NodeIndex::from(n2));
                        }
                        nodes.push(NodeIndex::from(n0));
                    } else {
                        panic!("node not found!");
                    }

                    continue;
                }

                if let Some(n2) = self.edge_data[*e].child_middle_node {
                    nodes.push(NodeIndex::from(n2));
                }

                if !nodes.contains(&NodeIndex::from(n0)) {
                    nodes.push(NodeIndex::from(n0));
                }
                if !nodes.contains(&NodeIndex::from(n1)) {
                    nodes.push(NodeIndex::from(n1));
                }
            }

            // add them
            mesh.add_face(&nodes, match self.face_data[f].boundary {Some(v) => Some(u16::from(v)), None => None}, self.face_data[f].owner, None);
            old_to_new_face_id[f] = Some(mesh.n_total_faces() - 1);
        }


        // now add the boundary patches in order
        for (_, bid) in &self.patch_info {
            let bid = *bid;

            for f in 0..self.face_data.len() {
                if self.face_data[f].refined_centernode.is_some() {
                    // this is a refined parent face, do not use
                    continue;
                }

                if !self.face_data[f].owner.owned() {
                    continue;
                }
                let Some(fbid) = self.face_data[f].boundary else {
                    continue;
                };
                if fbid != bid {continue};

                // this face is not a refined parent face
                // we can add it

                // get its nodes in order
                let mut nodes: Vec<NodeIndex> = vec![];
                let nodes_e1 = self.edge_nodes[self.face_edges.major_range(f)[self.face_edges.major_range(f).len() - 1]];
                for e in self.face_edges.major_range(f) {
                    let (n0, n1) = self.edge_nodes[*e];

                    if nodes.len() == 0 {
                        // add only the first one, the node shared by the second edge
                        if (n0 == nodes_e1.0) || (n0 == nodes_e1.1) {
                            nodes.push(NodeIndex::from(n0));
                            if let Some(n2) = self.edge_data[*e].child_middle_node {
                                nodes.push(NodeIndex::from(n2));
                            }
                            nodes.push(NodeIndex::from(n1));
                        } else if (n1 == nodes_e1.0) || (n1 == nodes_e1.1) {
                            nodes.push(NodeIndex::from(n1));
                            if let Some(n2) = self.edge_data[*e].child_middle_node {
                                nodes.push(NodeIndex::from(n2));
                            }
                            nodes.push(NodeIndex::from(n0));
                        } else {
                            panic!("node not found!");
                        }
                        continue;
                    }

                    if let Some(n2) = self.edge_data[*e].child_middle_node {
                        nodes.push(NodeIndex::from(n2));
                    }

                    if !nodes.contains(&NodeIndex::from(n0)) {
                        nodes.push(NodeIndex::from(n0));
                    }
                    if !nodes.contains(&NodeIndex::from(n1)) {
                        nodes.push(NodeIndex::from(n1));
                    }
                }

                // add them
                mesh.add_face(&nodes, match self.face_data[f].boundary {Some(v) => Some(u16::from(v)), None => None}, self.face_data[f].owner, None);
                old_to_new_face_id[f] = Some(mesh.n_total_faces() - 1);
            }
        }

        // now add the cells
        for c in 0..self.cell_data.len() {
            // if this cell is refined, skip it
            if self.cell_data[c].refined_cstart.is_some() {
                continue;
            }

            let mut faces: Vec<FaceIndex> = vec![];
            for f in self.cell_faces.major_range(c) {

                if self.face_data[*f].refined_fstart.is_some() {
                    // this face was refined, add its subfaces instead
                    let fsstart = self.face_data[*f].refined_fstart.unwrap();
                    let fssize = self.face_data[*f].refined_size.unwrap();

                    for fsub in fsstart..(fsstart+fssize) {
                        let f = old_to_new_face_id[fsub].unwrap();
                        faces.push(FaceIndex::from(f));
                    }
                } else {
                    // not a refined face, add it directly
                    let f = old_to_new_face_id[*f].unwrap();
                    faces.push(FaceIndex::from(f));
                }
            }

            // add its face
            mesh.add_cell(&faces, self.cell_data[c].owner, None);
        }

        // compute the mesh
        mesh.compute().unwrap();

        mesh
    }

}


