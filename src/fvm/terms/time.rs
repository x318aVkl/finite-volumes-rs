use crate::{fvm::{schemes::time::TimeScheme, terms::{Term, TermWrapper}}, prelude::{Unit, Zero}};




pub struct Time<'a, V, Lhs, const DIM: usize> {
    scheme: Box<dyn TimeScheme<DIM, Lhs=Lhs, Rhs=V> + 'a>,
}


impl<'a, V, Lhs, const DIM: usize> Time<'a, V, Lhs, DIM> {
    pub fn new(scheme: impl TimeScheme<DIM, Lhs=Lhs, Rhs=V> + 'a) -> Self {
        Self { scheme: Box::new(scheme) }
    }
}



pub fn time<'a, V, Lhs, const DIM: usize>(scheme: impl TimeScheme<DIM, Lhs=Lhs, Rhs=V> + 'a) -> TermWrapper<Time<'a, V, Lhs, DIM>, DIM>
where Lhs: Unit + Zero, V: Zero
{
    Time::new(scheme).wrap()
}



impl<'b, V, Lhs, const DIM: usize> Term<DIM> for Time<'b, V, Lhs, DIM> where Lhs: Unit + Zero, V: Zero {
    type Lhs = Lhs;
    type Rhs = V;

    fn cell_terms<'a>(&'a self, cell: &'a crate::prelude::CellRef<'a, DIM>, mesh: &'a crate::Mesh<DIM>) -> (Self::Lhs, Self::Rhs) {
        let (l, r) = self.scheme.terms(cell, mesh);
        (l, r)
    }

}



