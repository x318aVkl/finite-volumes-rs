use std::{marker::PhantomData, ops::{Add, Neg, Sub}};

use crate::{fvm::terms::{Term, TermWrapper}, prelude::Zero};



pub trait BinaryOp<A, B> {
    type Output;
    fn binary_op_eval(a: A, b: B) -> Self::Output;
}


pub struct TermsBinaryOp<LOP, ROP, A, B, Lhs, Rhs> {
    a: A,
    b: B,
    lop: PhantomData<LOP>,
    rop: PhantomData<ROP>,
    pdl: PhantomData<Lhs>,
    pdr: PhantomData<Rhs>,
}


impl<A, B, LOP, ROP, Lhs, Rhs, const DIM: usize> Term<DIM> for TermsBinaryOp<LOP, ROP, A, B, Lhs, Rhs>
    where A: Term<DIM, Lhs = Lhs, Rhs = Rhs>,
    B: Term<DIM, Lhs = Lhs, Rhs = Rhs>,
    LOP: BinaryOp<Lhs, Lhs, Output = Lhs>,
    ROP: BinaryOp<Rhs, Rhs, Output = Rhs>,
    Lhs: Zero,
    Rhs: Zero,
{
    type Lhs = Lhs;
    type Rhs = Rhs;

    fn cell_terms<'a>(&self, cell: &'a crate::prelude::CellRef<'a, DIM>, mesh: &'a crate::Mesh<DIM>) -> (Self::Lhs, Self::Rhs) {
        let ta = self.a.cell_terms(cell, mesh);
        let tb = self.b.cell_terms(cell, mesh);
        (
            LOP::binary_op_eval(ta.0, tb.0),
            ROP::binary_op_eval(ta.1, tb.1),
        )
    }

    fn face_terms<'a>(&self, face: &'a crate::prelude::FaceRef<'a, DIM>, mesh: &'a crate::Mesh<DIM>) -> (Self::Lhs, Self::Lhs, Self::Rhs) {
        let ta = self.a.face_terms(face, mesh);
        let tb = self.b.face_terms(face, mesh);
        (
            LOP::binary_op_eval(ta.0, tb.0),
            LOP::binary_op_eval(ta.1, tb.1),
            ROP::binary_op_eval(ta.2, tb.2),
        )
    }
}



pub struct BinaryOpAdd<A, B> {
    ap: PhantomData<A>,
    bp: PhantomData<B>,
}

impl<A, B> BinaryOp<A, B> for BinaryOpAdd<A, B>
where A: Add<B>
{
    type Output = <A as Add<B>>::Output;
    fn binary_op_eval(a: A, b: B) -> Self::Output {
        a + b
    }
}


pub struct BinaryOpSub<A, B> {
    ap: PhantomData<A>,
    bp: PhantomData<B>,
}

impl<A, B> BinaryOp<A, B> for BinaryOpSub<A, B>
where A: Sub<B>
{
    type Output = <A as Sub<B>>::Output;
    fn binary_op_eval(a: A, b: B) -> Self::Output {
        a - b
    }
}


impl<A, B, const DIM: usize> Add<TermWrapper<B, DIM>> for TermWrapper<A, DIM> 
where 
A: Term<DIM>, 
<A as Term<DIM>>::Lhs: Add<<A as Term<DIM>>::Lhs, Output = <A as Term<DIM>>::Lhs> + Copy,
<A as Term<DIM>>::Rhs: Add<<A as Term<DIM>>::Rhs, Output = <A as Term<DIM>>::Rhs> + Copy,
B: Term<DIM, Lhs = <A as Term<DIM>>::Lhs, Rhs = <A as Term<DIM>>::Rhs>,
{
    type Output = TermWrapper<
        TermsBinaryOp<
            BinaryOpAdd<<A as Term<DIM>>::Lhs, <A as Term<DIM>>::Lhs>,
            BinaryOpAdd<<A as Term<DIM>>::Rhs, <A as Term<DIM>>::Rhs>,
            TermWrapper<A, DIM>, 
            TermWrapper<B, DIM>, 
            <A as Term<DIM>>::Lhs, 
            <A as Term<DIM>>::Rhs
        >,
        DIM
    >;

    fn add(self, rhs: TermWrapper<B, DIM>) -> Self::Output {
        TermWrapper::new(
        TermsBinaryOp {
                a: self,
                b: rhs,
                lop: PhantomData,
                rop: PhantomData,
                pdl: PhantomData,
                pdr: PhantomData
            }
        )
    }
}



impl<A, B, const DIM: usize> Sub<TermWrapper<B, DIM>> for TermWrapper<A, DIM> 
where 
A: Term<DIM>, 
<A as Term<DIM>>::Lhs: Sub<<A as Term<DIM>>::Lhs, Output = <A as Term<DIM>>::Lhs> + Copy,
<A as Term<DIM>>::Rhs: Sub<<A as Term<DIM>>::Rhs, Output = <A as Term<DIM>>::Rhs> + Copy,
B: Term<DIM, Lhs = <A as Term<DIM>>::Lhs, Rhs = <A as Term<DIM>>::Rhs>,
{
    type Output = TermWrapper<
        TermsBinaryOp<
            BinaryOpSub<<A as Term<DIM>>::Lhs, <A as Term<DIM>>::Lhs>,
            BinaryOpSub<<A as Term<DIM>>::Rhs, <A as Term<DIM>>::Rhs>,
            TermWrapper<A, DIM>, 
            TermWrapper<B, DIM>, 
            <A as Term<DIM>>::Lhs, 
            <A as Term<DIM>>::Rhs
        >,
        DIM
    >;

    fn sub(self, rhs: TermWrapper<B, DIM>) -> Self::Output {
        TermWrapper::new(
        TermsBinaryOp {
                a: self,
                b: rhs,
                lop: PhantomData,
                rop: PhantomData,
                pdl: PhantomData,
                pdr: PhantomData
            }
        )
    }
}



pub struct OpNeg<A> {
    a: A,
}



impl<A, const DIM: usize> Term<DIM> for OpNeg<A> 
where A: Term<DIM>, 
<A as Term<DIM>>::Lhs: Neg<Output = <A as Term<DIM>>::Lhs>, 
<A as Term<DIM>>::Rhs: Neg<Output = <A as Term<DIM>>::Rhs>
{
    type Lhs = <A as Term<DIM>>::Lhs;
    type Rhs = <A as Term<DIM>>::Rhs;

    fn cell_terms<'a>(&self, cell: &'a crate::prelude::CellRef<'a, DIM>, mesh: &'a crate::Mesh<DIM>) -> (Self::Lhs, Self::Rhs) {
        let ta = self.a.cell_terms(cell, mesh);
        (
            - ta.0,
            - ta.1,
        )
    }

    fn face_terms<'a>(&self, face: &'a crate::prelude::FaceRef<'a, DIM>, mesh: &'a crate::Mesh<DIM>) -> (Self::Lhs, Self::Lhs, Self::Rhs) {
        let ta = self.a.face_terms(face, mesh);
        (
            - ta.0,
            - ta.1,
            - ta.2,
        )
    }

}

impl<A, const DIM: usize> Neg for TermWrapper<A, DIM> 
where A: Term<DIM>,
<A as Term<DIM>>::Lhs: Neg<Output = <A as Term<DIM>>::Lhs>, 
<A as Term<DIM>>::Rhs: Neg<Output = <A as Term<DIM>>::Rhs>
{
    type Output = TermWrapper<OpNeg<A>, DIM>;

    fn neg(self) -> Self::Output {
        TermWrapper::new(OpNeg {a: self.term})
    }
}

