use super::{GlobalRelation, Mesh, CellRef, FaceRef, FaceIndex, CellIndex, MeshGet};





pub trait Geometry<const DIM: usize>  {
    type IndexType;
    type ElementType<'a>: GlobalRelation;
    fn size_from_mesh(mesh: &Mesh<DIM>) -> usize;
    fn global_size_from_mesh(mesh: &Mesh<DIM>) -> usize;
    fn get_from_mesh<'a>(mesh: &'a Mesh<DIM>, index: Self::IndexType) -> Self::ElementType<'a>; 

    fn partially_owned_size_from_mesh(mesh: &Mesh<DIM>) -> usize;
}


#[derive(Clone, Copy, Debug)]
pub struct Face {}

impl<const DIM: usize> Geometry<DIM> for Face {
    type ElementType<'a> = FaceRef<'a, DIM>;
    type IndexType = FaceIndex;
    fn size_from_mesh(mesh: &Mesh<DIM>) -> usize {
        mesh.n_faces()
    }
    fn global_size_from_mesh(mesh: &Mesh<DIM>) -> usize {
        mesh.n_total_faces()
    }
    fn get_from_mesh<'a>(mesh: &'a Mesh<DIM>, index: Self::IndexType) -> Self::ElementType<'a> {
        mesh.get(index)
    }
    fn partially_owned_size_from_mesh(mesh: &Mesh<DIM>) -> usize {
        mesh.n_partially_local_faces()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Cell {}

impl<const DIM: usize> Geometry<DIM> for Cell {
    type ElementType<'a> = CellRef<'a, DIM>;
    type IndexType = CellIndex;
    fn size_from_mesh(mesh: &Mesh<DIM>) -> usize {
        mesh.n_cells()
    }
    fn global_size_from_mesh(mesh: &Mesh<DIM>) -> usize {
        mesh.n_total_cells()
    }
    fn get_from_mesh<'a>(mesh: &'a Mesh<DIM>, index: Self::IndexType) -> Self::ElementType<'a> {
        mesh.get(index)
    }
    fn partially_owned_size_from_mesh(mesh: &Mesh<DIM>) -> usize {
        mesh.n_cells()
    }
}

