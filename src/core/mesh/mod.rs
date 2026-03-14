




mod vtk;
pub mod io;
pub mod decompose;
pub mod geometry;
pub mod compute;
pub mod examples;

use std::usize;

pub use geometry::Geometry;


use mpi::{topology::SimpleCommunicator, traits::Communicator};

use crate::core::{Sparsity, Vector, communicator::SingleDataCommunicator};


#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NodeIndex(usize);

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FaceIndex(usize);

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CellIndex(usize);


impl From<NodeIndex> for usize {
    fn from(value: NodeIndex) -> Self {
        value.0
    }
}
impl From<FaceIndex> for usize {
    fn from(value: FaceIndex) -> Self {
        value.0
    }
}

impl From<CellIndex> for usize {
    fn from(value: CellIndex) -> Self {
        value.0
    }
}


impl From<usize> for NodeIndex {
    fn from(value: usize) -> Self {
        Self(value)
    }
}


impl From<usize> for FaceIndex {
    fn from(value: usize) -> Self {
        Self(value)
    }
}

impl From<usize> for CellIndex {
    fn from(value: usize) -> Self {
        Self(value)
    }
}

impl std::ops::Add for NodeIndex {
    type Output = Self;
    fn add(mut self, rhs: Self) -> Self::Output {
        self.0 += rhs.0;
        self
    }
}
impl std::ops::Add for FaceIndex {
    type Output = Self;
    fn add(mut self, rhs: Self) -> Self::Output {
        self.0 += rhs.0;
        self
    }
}
impl std::ops::Add for CellIndex {
    type Output = Self;
    fn add(mut self, rhs: Self) -> Self::Output {
        self.0 += rhs.0;
        self
    }
}
impl std::fmt::Display for NodeIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Node({})", self.0)
    }
}
impl std::fmt::Display for FaceIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Face({})", self.0)
    }
}
impl std::fmt::Display for CellIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Cell({})", self.0)
    }
}


#[derive(Clone, Copy, Debug)]
pub enum FaceNeighbor {
    Boundary(u16),
    Cell(CellIndex),
}
#[derive(Clone, Copy, Debug)]
enum InternalFaceNeighbor {
    Boundary(u16),
    Cell(CellIndex),
    None,
}

#[derive(Clone, Copy, Debug)]
pub struct CellData<const DIM: usize> {
    volume: f64,
    center: Vector<DIM>,
    ownership: Ownership,
    global_id: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct FaceData<const DIM: usize> {
    area: f64,
    center: Vector<DIM>,
    normal: Vector<DIM>,
    owner_cell: CellIndex,
    neighbor: InternalFaceNeighbor,
    ownership: Ownership,
    global_id: u32,
}


#[derive(Clone, Copy)]
pub struct NodeRef<'a, const DIM: usize> {
    id: NodeIndex,
    mesh: &'a Mesh<DIM>,
}

#[derive(Clone, Copy)]
pub struct CellRef<'a, const DIM: usize> {
    id: CellIndex,
    data: &'a CellData<DIM>,
    faces: &'a [FaceIndex],
    mesh: &'a Mesh<DIM>,
}

#[derive(Clone, Copy)]
pub struct FaceRef<'a, const DIM: usize> {
    id: FaceIndex,
    data: &'a FaceData<DIM>,
    nodes: &'a [NodeIndex],
    mesh: &'a Mesh<DIM>,
}

#[derive(Clone, Copy, Debug)]
pub enum Ownership {
    Owned,
    Remote(usize),
}

impl Ownership {
    pub fn owned(&self) -> bool {
        match self {
            Self::Owned => true,
            _ => false,
        }
    }
}



impl<'a, const DIM: usize> std::fmt::Debug for CellRef<'a, DIM> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Cell(id={}, volume={}, center={:?})", self.id().0, self.volume(), self.center())
    }
}


/// Represents a mesh for mixed fvm-fem representation
pub struct Mesh<const DIM: usize> {

    nodes: Vec<Vector<DIM>>,

    face_nodes: Sparsity<NodeIndex>,
    cell_faces: Sparsity<FaceIndex>,

    // cell to cell connectivity, includes own cell
    cell_to_cell: Sparsity<CellIndex>,

    /// One vector per connected cell for each cell
    cell_face_gradient_coefficients: Vec<Vector<DIM>>,
    cell_diag_gradient_coefficients: Vec<Vector<DIM>>,

    face_data: Vec<FaceData<DIM>>,
    cell_data: Vec<CellData<DIM>>,

    face_boundaries: Vec<Option<u16>>,

    n_local_faces: usize,
    n_local_cells: usize,

    computed: bool,

    mpi_comm: Option<SimpleCommunicator>,

}


impl<'a, const DIM: usize> NodeRef<'a, DIM> {
    pub fn id(&self) -> NodeIndex {
        self.id
    }
    pub fn position(&self) -> Vector<DIM> {
        self.mesh.nodes[usize::from(self.id)]
    }
    pub fn center(&self) -> Vector<DIM> {
        self.position()
    }
}


impl<'a, const DIM: usize> FaceRef<'a, DIM> {
    pub fn id(&self) -> FaceIndex {
        self.id
    }
    pub fn center(&self) -> Vector<DIM> {
        self.data.center
    }
    pub fn area(&self) -> f64 {
        self.data.area
    }
    pub fn normal(&self) -> Vector<DIM> {
        self.data.normal
    }
    pub fn boundary(&self) -> Option<u16> {
        self.mesh.face_boundaries[usize::from(self.id)]
    }
    pub fn nodes(&self) -> &[NodeIndex] {
        self.nodes
    }
    pub fn n_nodes(&self) -> usize {
        self.nodes.len()
    }
    pub fn node(&self, node: NodeIndex) -> NodeRef<'a, DIM> {
        self.mesh.node(node)
    }
    pub fn outer_normal(&self, c: Vector<DIM>) -> Vector<DIM> {
        let n = self.data.normal;
        let dfc = self.data.center - c;
        let s = dfc.dot(n);
        let s = s / s.abs();
        n * s
    }
    pub fn global_id(&self) -> u32 {
        self.data.global_id
    }
    pub fn ownership(&self) -> Ownership {
        self.data.ownership
    }
    pub fn other_cell(&self, cell: CellIndex) -> Option<CellIndex> {
        match self.data.neighbor {
            InternalFaceNeighbor::Boundary(_) => None,
            InternalFaceNeighbor::None => None,
            InternalFaceNeighbor::Cell(c1) => {
                if cell == c1 {
                    Some(self.data.owner_cell)
                } else {
                    Some(c1)
                }
            }
        }
    }
    pub fn fully_remote(&self) -> bool {
        let owner_remote = !self.mesh.cell(self.data.owner_cell).owned();
        match self.data.neighbor {
            InternalFaceNeighbor::Cell(c) => {
                let neighbor_remote = !self.mesh.cell(c).owned();
                owner_remote & neighbor_remote
            },
            _ => {
                owner_remote
            }
        }
    }
    pub fn owner(&self) -> CellIndex {
        self.data.owner_cell
    }
    pub fn neighbor(&self) -> FaceNeighbor {
        match self.data.neighbor {
            InternalFaceNeighbor::Cell(c) => FaceNeighbor::Cell(c),
            InternalFaceNeighbor::Boundary(b) => FaceNeighbor::Boundary(b),
            InternalFaceNeighbor::None => panic!("face neighbor is none for face {}", self.id)
        }
    }
}

impl<'a, const DIM: usize> CellRef<'a, DIM> {
    pub fn id(&self) -> CellIndex {
        self.id
    }
    pub fn center(&self) -> Vector<DIM> {
        self.data.center
    }
    pub fn volume(&self) -> f64 {
        self.data.volume
    }
    pub fn faces(&self) -> &[FaceIndex] {
        self.faces
    }
    pub fn n_faces(&self) -> usize {
        self.faces.len()
    }
    pub fn node(&self, node: NodeIndex) -> NodeRef<'a, DIM> {
        self.mesh.node(node)
    }
    pub fn iter_grad(&self) -> impl Iterator<Item = (FaceIndex, Vector<DIM>)> {
        (self.mesh.cell_faces.major_start(usize::from(self.id))..self.mesh.cell_faces.major_end(usize::from(self.id)))
        .map(|k| {
            (
                self.mesh.cell_faces.flat_index(k),
                self.mesh.cell_face_gradient_coefficients[k]
            )
        })
    }
    pub fn own_grad(&self) -> Vector<DIM> {
        self.mesh.cell_diag_gradient_coefficients[usize::from(self.id)]
    }
    pub fn global_id(&self) -> u32 {
        self.data.global_id
    }
    pub fn ownership(&self) -> Ownership {
        self.data.ownership
    }
    pub fn owned(&self) -> bool {
        match self.data.ownership {
            Ownership::Owned => true,
            _ => false,
        }
    }
}


impl<const DIM: usize> Mesh<DIM> {

    pub fn new(mpi_comm: Option<SimpleCommunicator>) -> Self {
        Self {
            nodes: vec![],
            face_nodes: Sparsity::new(),
            cell_faces: Sparsity::new(),
            cell_to_cell: Sparsity::new(),
            cell_face_gradient_coefficients: vec![],
            cell_diag_gradient_coefficients: vec![],
            face_data: vec![],
            cell_data: vec![],
            face_boundaries: vec![],
            n_local_faces: 0,
            n_local_cells: 0,
            computed: false,
            mpi_comm,
        }
    }

    pub fn add_node(&mut self, node: Vector<DIM>) {
        self.computed = false;

        self.nodes.push(node);

    }

    pub fn add_face(&mut self, nodes: &[NodeIndex], boundary: Option<u16>, ownership: Ownership, global_id: Option<u32>) {
        self.computed = false;

        for n in nodes {
            self.face_nodes.push_to_major(*n);
        }
        self.face_nodes.close_major();

        self.face_boundaries.push(boundary);

        let fid = self.face_data.iter().len() as u32;

        match ownership {
            Ownership::Owned => self.n_local_faces += 1,
            _ => {}
        }

        self.face_data.push(FaceData { 
            area: 0.0, 
            center: Vector::new(), 
            normal: Vector::new(), 
            owner_cell: CellIndex(usize::MAX),
            neighbor: match boundary {Some(i) => InternalFaceNeighbor::Boundary(i), None => InternalFaceNeighbor::None},
            ownership, 
            global_id: match global_id {
                Some(v) => v,
                None => fid,
            } 
        })
    }

    pub fn add_cell(&mut self, faces: &[FaceIndex], ownership: Ownership, global_id: Option<u32>) {
        self.computed = false;

        for n in faces {
            self.cell_faces.push_to_major(*n);
        }
        self.cell_faces.close_major();

        let cid = self.cell_data.iter().len() as u32;

        match ownership {
            Ownership::Owned => self.n_local_cells += 1,
            _ => {}
        }

        self.cell_data.push(CellData { 
            volume: 0.0, 
            center: Vector::new(), 
            ownership, 
            global_id: match global_id {
                Some(v) => v,
                None => cid,
            } 
        });

        // set the face owner or neighbor to be this cell
        for f in faces {
            let fi = usize::from(*f);
            if usize::from(self.face_data[fi].owner_cell) == usize::MAX {
                self.face_data[fi].owner_cell = CellIndex(self.cell_data.len() - 1);
            } else {
                match self.face_data[fi].neighbor {
                    InternalFaceNeighbor::None => {
                        self.face_data[fi].neighbor = InternalFaceNeighbor::Cell(CellIndex(self.cell_data.len() - 1));
                    },
                    _ => {
                        panic!("Error, trying to add another cell to a face with neighbor: {:?}", self.face_data[fi].neighbor);
                    }
                }
            }
        }
    }

    pub fn n_nodes(&self) -> usize {
        self.nodes.len()
    }

    pub fn n_total_nodes(&self) -> usize {
        self.nodes.len()
    }

    pub fn n_faces(&self) -> usize {
        self.n_local_faces
    }

    pub fn n_total_faces(&self) -> usize {
        self.face_nodes.major_len()
    }

    pub fn n_cells(&self) -> usize {
        self.n_local_cells
    }

    pub fn n_total_cells(&self) -> usize {
        self.cell_faces.major_len()
    }

    pub fn node<'a>(&'a self, node: NodeIndex) -> NodeRef<'a, DIM> {
        NodeRef {
            id: node,
            mesh: self,
        }
    }

    pub fn cell<'a>(&'a self, cell: CellIndex) -> CellRef<'a, DIM> {
        CellRef {
            id: cell,
            data: &self.cell_data[usize::from(cell)],
            faces: self.cell_faces.major_range(usize::from(cell)),
            mesh: &self,
        }
    }

    pub fn face<'a>(&'a self, face: FaceIndex) -> FaceRef<'a, DIM> {
        FaceRef {
            id: face,
            data: &self.face_data[usize::from(face)],
            nodes: self.face_nodes.major_range(usize::from(face)),
            mesh: &self,
        }
    }

    pub fn iter_nodes<'a>(&'a self) -> NodeIterator<'a, DIM> {
        NodeIterator { current: 0, mesh: self, }
    }

    pub fn iter_faces<'a>(&'a self) -> FaceIterator<'a, DIM> {
        FaceIterator { current: 0, mesh: self, skip_remote: true, }
    }

    pub fn iter_cells<'a>(&'a self) -> CellIterator<'a, DIM> {
        CellIterator { current: 0, mesh: self, skip_remote: true }
    }

    pub fn iter_all_nodes<'a>(&'a self) -> NodeIterator<'a, DIM> {
        NodeIterator { current: 0, mesh: self, }
    }

    pub fn iter_all_faces<'a>(&'a self) -> FaceIterator<'a, DIM> {
        FaceIterator { current: 0, mesh: self, skip_remote: false, }
    }

    pub fn iter_all_cells<'a>(&'a self) -> CellIterator<'a, DIM> {
        CellIterator { current: 0, mesh: self, skip_remote: false }
    }

    pub fn communicator<'a>(&'a self) -> Option<&'a SimpleCommunicator> {
        self.mpi_comm.as_ref()
    }

    pub fn communicator_clone(&self) -> Option<SimpleCommunicator> {
        match &self.mpi_comm {
            Some(v) => Some(v.duplicate()),
            None => None,
        }
    }

    pub fn comm<'a>(&'a self) -> SingleDataCommunicator<'a> {
        SingleDataCommunicator::from_mpi_comm(self.mpi_comm.as_ref())
    }

    pub fn cell_to_cell_sparsity(&self) -> &Sparsity<CellIndex> {
        &self.cell_to_cell
    }

}



pub trait MeshGet<'a, Index> {
    type Output;
    fn get(&'a self, index: Index) -> Self::Output;
}

impl<'a, const DIM: usize> MeshGet<'a, NodeIndex> for Mesh<DIM> {
    type Output = NodeRef<'a, DIM>;
    fn get(&'a self, index: NodeIndex) -> Self::Output {
        self.node(index)
    }
}
impl<'a, const DIM: usize> MeshGet<'a, FaceIndex> for Mesh<DIM> {
    type Output = FaceRef<'a, DIM>;
    fn get(&'a self, index: FaceIndex) -> Self::Output {
        self.face(index)
    }
}
impl<'a, const DIM: usize> MeshGet<'a, CellIndex> for Mesh<DIM> {
    type Output = CellRef<'a, DIM>;
    fn get(&'a self, index: CellIndex) -> Self::Output {
        self.cell(index)
    }
}


pub trait GlobalRelation {
    fn ownership(&self) -> Ownership;
    fn local_id(&self) -> usize;
    fn global_id(&self) -> u32;
}


impl<'a, const DIM: usize> GlobalRelation for FaceRef<'a, DIM> {
    fn local_id(&self) -> usize {
        usize::from(self.id)
    }
    fn global_id(&self) -> u32 {
        self.global_id()
    }
    fn ownership(&self) -> Ownership {
        self.ownership()
    }
}

impl<'a, const DIM: usize> GlobalRelation for CellRef<'a, DIM> {
    fn local_id(&self) -> usize {
        usize::from(self.id)
    }
    fn global_id(&self) -> u32 {
        self.global_id()
    }
    fn ownership(&self) -> Ownership {
        self.ownership()
    }
}




pub struct NodeIterator<'a, const DIM: usize> {
    current: usize,
    mesh: &'a Mesh<DIM>,
}

impl<'a, const DIM: usize> Iterator for NodeIterator<'a, DIM> {
    type Item = NodeRef<'a, DIM>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.current >= self.mesh.n_total_nodes() {
            None
        } else {
            let out = 
                self.mesh.node(self.current.into())
            ;
            self.current += 1;
            Some(out)
        }
    }
}


pub struct FaceIterator<'a, const DIM: usize> {
    current: usize,
    mesh: &'a Mesh<DIM>,
    skip_remote: bool,
}

impl<'a, const DIM: usize> Iterator for FaceIterator<'a, DIM> {
    type Item = FaceRef<'a, DIM>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.current >= self.mesh.n_total_faces() {
            None
        } else if self.skip_remote && self.mesh.face(self.current.into()).fully_remote() {
            self.current += 1;
            self.next()
        } else {
            let out = 
                self.mesh.face(self.current.into())
            ;
            self.current += 1;
            Some(out)
        }
    }
}

pub struct CellIterator<'a, const DIM: usize> {
    current: usize,
    mesh: &'a Mesh<DIM>,
    skip_remote: bool,
}

impl<'a, const DIM: usize> Iterator for CellIterator<'a, DIM> {
    type Item = CellRef<'a, DIM>;
    fn next(&mut self) -> Option<Self::Item> {
        if (self.current >= self.mesh.n_total_cells()) || (self.skip_remote && (!self.mesh.cell(self.current.into()).ownership().owned())) {
            None
        } else {
            let out = 
                self.mesh.cell(self.current.into())
            ;
            self.current += 1;
            Some(out)
        }
    }
}
