use std::marker::PhantomData;

use mpi::traits::Equivalence;

use crate::{core::{Mesh, communicator::Communicator, mesh::{BoundaryPatch, Geometry, MeshGet, PatchIndex}}, prelude::{FaceIndex, geometry}};




#[derive(Clone)]
pub struct Field<T, G: Geometry<DIM>, const DIM: usize> {
    data: Vec<T>,
    local_len: usize,
    comm: Communicator<G, DIM>,
    _phantom: PhantomData<G>,
}



impl<'a, I, T: Default + Clone, G: Geometry<DIM, IndexType = I>, const DIM: usize> Field<T, G, DIM>
where T: Equivalence, Mesh<DIM>: MeshGet<'a, I>, I: From<usize>, usize: From<I> {
    pub fn from_mesh(mesh: &Mesh<DIM>) -> Self {
        Field {
            data: vec![T::default(); G::global_size_from_mesh(mesh)],
            local_len: G::size_from_mesh(mesh),
            comm: Communicator::from_mesh(&mesh),
            _phantom: PhantomData,
        }
    }

    pub fn update(&mut self) {
        self.comm.collect(&mut self.data);
    }

    pub fn len(&self) -> usize {
        self.local_len
    }

    pub fn total_len(&self) -> usize {
        self.data.len()
    }

    pub fn raw_data(&self) -> &[T] {
        &self.data
    }

    pub fn set_from(&mut self, data: &[T]) where T: Copy {
        assert!((self.total_len() == data.len()) || (self.len() == data.len()));
        for i in 0..data.len() {
            self.data[i] = data[i];
        }
        self.update();
    }
}


impl<T, G: Geometry<DIM>, const DIM: usize> std::ops::Index<G::IndexType> for Field<T, G, DIM> where usize: From<<G as Geometry<DIM>>::IndexType> {
    type Output = T;
    fn index(&self, index: G::IndexType) -> &Self::Output {
        &self.data[usize::from(index)]
    }
}


impl<T, G: Geometry<DIM>, const DIM: usize> std::ops::IndexMut<G::IndexType> for Field<T, G, DIM> where usize: From<<G as Geometry<DIM>>::IndexType> {
    fn index_mut(&mut self, index: G::IndexType) -> &mut Self::Output {
        &mut self.data[usize::from(index)]
    }
}



pub struct PatchField<T, const DIM: usize> {
    data: Vec<T>,
    patch_id: PatchIndex,
    fstart: FaceIndex,
}


impl<T, const DIM: usize> PatchField<T, DIM> {

    pub fn from_mesh(mesh: &Mesh<DIM>, patch: PatchIndex) -> Self where T: Default + Copy {
        let patch = mesh.patch(patch);
        Self {
            data: vec![T::default(); patch.len()],
            patch_id: patch.id(),
            fstart: patch.face_start(),
        }
    }

    pub fn with_constant(mut self, value: T) -> Self where T: Copy {
        for i in 0..self.data.len() {
            self.data[i] = value;
        }
        self
    }

    pub fn set_map<'a>(mut self, fun: impl Fn(FaceIndex) -> T) -> Self {
        for i in 0..self.len() {
            let f = self.fstart + FaceIndex::from(i);
            self.data[i] = fun(f);
        }
        self
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn id(&self) -> PatchIndex {
        self.patch_id
    }

}


impl<T, const DIM: usize> std::ops::Index<FaceIndex> for PatchField<T, DIM> {
    type Output = T;
    fn index(&self, index: FaceIndex) -> &Self::Output {
        let f = self.fstart - index;
        let f = usize::from(f);
        assert!(f < self.len());
        &self.data[f]
    }
}
impl<T, const DIM: usize> std::ops::IndexMut<FaceIndex> for PatchField<T, DIM> {
    fn index_mut(&mut self, index: FaceIndex) -> &mut Self::Output {
        let f = self.fstart - index;
        let f = usize::from(f);
        assert!(f < self.len());
        &mut self.data[f]
    }
}


impl<'a, T, const DIM: usize> Into<PatchFieldView<'a, T, DIM>> for &'a PatchField<T, DIM> {
    fn into(self) -> PatchFieldView<'a, T, DIM> {
        PatchFieldView { data: &self.data, patch_id: self.patch_id, fstart: self.fstart }
    }
}
impl<'a, T, const DIM: usize> Into<PatchFieldViewMut<'a, T, DIM>> for &'a mut PatchField<T, DIM> {
    fn into(self) -> PatchFieldViewMut<'a, T, DIM> {
        PatchFieldViewMut { data: &mut self.data, patch_id: self.patch_id, fstart: self.fstart }
    }
}




pub struct PatchFieldView<'a, T, const DIM: usize> {
    data: &'a [T],
    patch_id: PatchIndex,
    fstart: FaceIndex,
}


impl<'a, T, const DIM: usize> PatchFieldView<'a, T, DIM> {

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn id(&self) -> PatchIndex {
        self.patch_id
    }

}

impl<'a, T, const DIM: usize> std::ops::Index<FaceIndex> for PatchFieldView<'a, T, DIM> {
    type Output = T;
    fn index(&self, index: FaceIndex) -> &Self::Output {
        let f = self.fstart - index;
        let f = usize::from(f);
        assert!(f < self.len());
        &self.data[f]
    }
}




pub struct PatchFieldViewMut<'a, T, const DIM: usize> {
    data: &'a mut [T],
    patch_id: PatchIndex,
    fstart: FaceIndex,
}


impl<'a, T, const DIM: usize> PatchFieldViewMut<'a, T, DIM> {

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn id(&self) -> PatchIndex {
        self.patch_id
    }

}

impl<'a, T, const DIM: usize> std::ops::Index<FaceIndex> for PatchFieldViewMut<'a, T, DIM> {
    type Output = T;
    fn index(&self, index: FaceIndex) -> &Self::Output {
        let f = self.fstart - index;
        let f = usize::from(f);
        assert!(f < self.len());
        &self.data[f]
    }
}




impl<'a, T, const DIM: usize> std::ops::IndexMut<FaceIndex> for PatchFieldViewMut<'a, T, DIM> {
    fn index_mut(&mut self, index: FaceIndex) -> &mut Self::Output {
        let f = self.fstart - index;
        let f = usize::from(f);
        assert!(f < self.len());
        &mut self.data[f]
    }
}


impl<T, const DIM: usize> Field<T, geometry::Face, DIM> {

    pub fn patch<'a, 'b>(&'a self, patch: &BoundaryPatch<'b, DIM>) -> PatchFieldView<'a, T, DIM> {
        let fstart = patch.face_start();
        let fus = usize::from(fstart);
        let len = patch.len();
        PatchFieldView {
            data: &self.data[fus..(fus + len)],
            patch_id: patch.id(),
            fstart: fstart,
        }
    }

    pub fn patch_mut<'a, 'b>(&'a mut self, patch: &BoundaryPatch<'b, DIM>) -> PatchFieldViewMut<'a, T, DIM> {
        let fstart = patch.face_start();
        let fus = usize::from(fstart);
        let len = patch.len();
        PatchFieldViewMut {
            data: &mut self.data[fus..(fus + len)],
            patch_id: patch.id(),
            fstart: fstart,
        }
    }

}


