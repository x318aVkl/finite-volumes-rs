use crate::{Mesh, core::mesh::{FaceIndex, FaceRef}};





pub struct FaceConstraints<T, Lhs> {
    values: Vec<(Lhs, T)>,
    face_start: usize,
}


impl<T, Lhs, const DIM: usize> From<&Mesh<DIM>> for FaceConstraints<T, Lhs> 
where T: Default + Clone, Lhs: Default + Clone
{
    fn from(value: &Mesh<DIM>) -> Self {
        Self::from_mesh(value)
    }
}

impl<T, Lhs> FaceConstraints<T, Lhs> {
    fn from_mesh<const DIM: usize>(mesh: &Mesh<DIM>) -> Self where T: Default + Clone, Lhs: Default + Clone {
        let fs = mesh.iter_patch().nth(0).unwrap().face_start();
        let fs = usize::from(fs);
        let plast = mesh.iter_patch().last().unwrap();
        let flen = (usize::from(plast.face_start()) + plast.len()) - fs;
        Self {
            values: vec![(Lhs::default(), T::default()); flen],
            face_start: fs,
        }
    }

    pub fn as_bc<const DIM: usize>(&self) -> impl Fn(&FaceRef<DIM>) -> (Lhs, T) where Lhs: Copy, T: Copy {
        |face| {
            self[face.id()]
        }
    }
}

impl<T, Lhs> std::ops::Index<FaceIndex> for FaceConstraints<T, Lhs> {
    type Output = (Lhs, T);
    fn index(&self, index: FaceIndex) -> &Self::Output {
        let f = usize::from(index);
        assert!(f >= self.face_start);
        let f = f - self.face_start;
        &self.values[f]
    }
}


impl<T, Lhs> std::ops::IndexMut<FaceIndex> for FaceConstraints<T, Lhs> {
    fn index_mut(&mut self, index: FaceIndex) -> &mut Self::Output {
        let f = usize::from(index);
        assert!(f >= self.face_start);
        let f = f - self.face_start;
        &mut self.values[f]
    }
}
