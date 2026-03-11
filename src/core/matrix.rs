

use mpi::traits::Equivalence;

use crate::core::{Vector, error::Error, traits::FloatBuffered};


/// Small f64 matrix of size M rows by N columns. Used to represent tensors, gradients of vector fields...
#[derive(Clone, Copy, Debug)]
pub struct Matrix<const M: usize, const N: usize> {
    rows: [Vector<N>; M]
}


impl<const M: usize, const N: usize> Matrix<M, N> {

    pub fn new() -> Matrix<M, N> {
        Matrix { rows: [Vector::new(); M] }
    }

    pub fn eye() -> Matrix<M, N> {
        let mut out = Matrix::new();

        for i in 0..M.min(N) {
            out[[i, i]] = 1.0;
        }

        out
    }

    pub fn diag<const D: usize>(d: Vector<D>) -> Matrix<D, D> {
        let mut out = Matrix::new();

        for i in 0..D {
            out[[i, i]] = d[i];
        }

        out
    }

    pub fn dot(self, rhs: Vector<N>) -> Vector<M> {
        self * rhs
    }

    pub fn row(&self, i: usize) -> Vector<N> {
        self.rows[i]
    }

    pub fn trace(&self) -> f64 {
        let mut out = 0.0;
        for i in 0..M.min(N) {
            out += self[[i, i]];
        }
        out
    }

    pub fn col(self, i: usize) -> Vector<M> {
        let mut col = Vector::<M>::new();
        for j in 0..M {
            col[j] = self[[j, i]];
        }
        col
    }

    pub fn set_col(&mut self, i: usize, col: Vector<M>) {
        for j in 0..M {
            self[[j, i]] = col[j];
        }
    }

    pub fn transpose(self) -> Matrix<N, M> {
        let mut out = Matrix::<N, M>::new();
        for i in 0..M {
            for j in 0..N {
                out[[j, i]] = self[[i, j]];
            }
        }        
        out
    }

}


impl<const N: usize> Vector<N> {
    pub fn outer<const M: usize>(self, rhs: Vector<M>) -> Matrix<N, M> {
        let mut out = Matrix::new();
        
        for i in 0..N {
            for j in 0..M {
                out[[i, j]] = self[i] * rhs[j];
            }
        }

        out
    }
}



impl<const M: usize, const N: usize> std::ops::Index<[usize; 2]> for Matrix<M, N> {
    type Output = f64;
    fn index(&self, index: [usize; 2]) -> &Self::Output {
        &self.rows[index[0]][index[1]]
    }
}

impl<const M: usize, const N: usize> std::ops::IndexMut<[usize; 2]> for Matrix<M, N> {
    fn index_mut(&mut self, index: [usize; 2]) -> &mut Self::Output {
        &mut self.rows[index[0]][index[1]]
    }
}



impl<const M: usize, const N: usize> std::ops::MulAssign<f64> for Matrix<M, N> {
    fn mul_assign(&mut self, rhs: f64) {
        for i in 0..M {
            for j in 0..N {
                self[[i, j]] *= rhs;
            }
        }
    }
}


impl<const M: usize, const N: usize> std::ops::DivAssign<f64> for Matrix<M, N> {
    fn div_assign(&mut self, rhs: f64) {
        for i in 0..M {
            for j in 0..N {
                self[[i, j]] /= rhs;
            }
        }
    }
}




impl<const M: usize, const N: usize> std::ops::AddAssign for Matrix<M, N> {
    fn add_assign(&mut self, rhs: Self) {
        for i in 0..M {
            for j in 0..N {
                self[[i, j]] += rhs[[i, j]];
            }
        }
    }
}


impl<const M: usize, const N: usize> std::ops::SubAssign for Matrix<M, N> {
    fn sub_assign(&mut self, rhs: Self) {
        for i in 0..M {
            for j in 0..N {
                self[[i, j]] -= rhs[[i, j]];
            }
        }
    }
}



impl<const M: usize, const N: usize> std::ops::Mul<f64> for Matrix<M, N> {
    type Output = Self;
    fn mul(mut self, rhs: f64) -> Self::Output {
        self *= rhs;
        self
    }
}



impl<const M: usize, const N: usize> std::ops::Div<f64> for Matrix<M, N> {
    type Output = Self;
    fn div(mut self, rhs: f64) -> Self::Output {
        self /= rhs;
        self
    }
}




impl<const M: usize, const N: usize> std::ops::Add for Matrix<M, N> {
    type Output = Self;
    fn add(mut self, rhs: Self) -> Self::Output {
        self += rhs;
        self
    }
}


impl<const M: usize, const N: usize> std::ops::Sub for Matrix<M, N> {
    type Output = Self;
    fn sub(mut self, rhs: Self) -> Self::Output {
        self -= rhs;
        self
    }
}


impl<const M: usize, const N: usize> std::ops::Mul<Vector<N>> for Matrix<M, N> {
    type Output = Vector<M>;
    fn mul(self, rhs: Vector<N>) -> Self::Output {
        let mut out = Vector::<M>::new();
        for i in 0..M {
            out[i] = self.rows[i].dot(rhs);
        }
        out
    }
}

impl<const M: usize, const N: usize> std::ops::Mul<Matrix<M, N>> for f64 {
    type Output = Matrix<M, N>;
    fn mul(self, rhs: Matrix<M, N>) -> Self::Output {
        rhs * self
    }
}

impl<const M: usize, const N: usize> std::ops::Neg for Matrix<M, N> {
    type Output = Self;
    fn neg(mut self) -> Self::Output {
        for i in 0..M {
            for j in 0..N {
                self[[i, j]] = -self[[i, j]];
            }
        }
        self
    }
}

impl<const M: usize, const N: usize, const K: usize> std::ops::Mul<Matrix<K, N>> for Matrix<M, K> {
    type Output = Matrix<M, N>;
    fn mul(self, rhs: Matrix<K, N>) -> Self::Output {
        let mut out = Matrix::new();
        
        for i in 0..M {
            for j in 0..N {
                for k in 0..K {
                    out[[i, j]] += self[[i, k]] * rhs[[k, j]];
                }
            }
        }

        out
    }
}




pub struct DynamicMatrix {
    data: Vec<f64>,
    cols: usize,
}


impl DynamicMatrix {
    pub fn new(m: usize, n: usize) -> Self {
        Self {
            data: vec![0.0; m*n],
            cols: n,
        }
    }
    pub fn nrows(&self) -> usize {
        self.data.len() / self.cols
    }
    pub fn ncols(&self) -> usize {
        self.cols
    }
}

impl std::ops::Index<[usize; 2]> for DynamicMatrix {
    type Output = f64;
    fn index(&self, index: [usize; 2]) -> &Self::Output {
        &self.data[index[0]*self.cols + index[1]]
    }
}

impl std::ops::IndexMut<[usize; 2]> for DynamicMatrix {
    fn index_mut(&mut self, index: [usize; 2]) -> &mut Self::Output {
        &mut self.data[index[0]*self.cols + index[1]]
    }
}


// square matrices decompositions and operations
impl<const N: usize> Matrix<N, N> {


    pub fn from_slice(data: &[f64]) -> Matrix<N, N> {
        let mut mat = Matrix::new();
        for i in 0..N {
            for j in 0..N {
                mat[[i, j]] = data[i*N + j];
            }
        }
        mat
    }

    pub fn inv(mut self) -> Result<Self, Error> {

        // compute the inverse of the matrix
        let mut out = Self::eye();

        let mut p: [usize; N] = [0; N];
        for i in 0..N {
            p[i] = i;
        }

        for k in 0..(N-1) {
            let piv = self[[k, k]];

            if piv.abs() < 1e-14 {
                // swap with largest
                let mut maxi = k;
                let mut maxp = piv.abs();
                for j in (k+1)..N {
                    let pj = self[[k, j]].abs();
                    if pj > maxp {
                        maxi = j;
                        maxp = pj;
                    }
                }
                if maxi == k {
                    return Err(Error::SingularMatrix);
                }
                // swap rows
                self.rows.swap(k, maxi);
                out.rows.swap(k, maxi);
                p.swap(k, maxi);
            }
            let piv = self[[k, k]];

            for i in (k+1)..N {
                let f = - self[[i, k]] / piv;

                self.rows[i] += self.rows[k] * f;
                out.rows[i] += out.rows[k] * f;
            }
        }

        for k in 0..N {
            let piv = self[[k, k]];
            if piv.abs() < 1e-14 {
                return Err(Error::SingularMatrix);
            }
            self.rows[k] /= piv;
            out.rows[k] /= piv;
        }

        for k in (1..N).rev() {
            let piv = self[[k, k]];

            for i in 0..k {
                let f = - self[[i, k]] / piv;
                self.rows[i] += self.rows[k] * f;
                out.rows[i] += out.rows[k] * f;
            }
        }

        Ok(out)
    }


        pub fn det(mut self) -> Result<f64, Error> {

        // compute reduce row echelon form
        let mut p: [usize; N] = [0; N];
        for i in 0..N {
            p[i] = i;
        }

        let mut pdet = 1.0;
        for k in 0..(N-1) {
            let piv = self[[k, k]];

            if piv.abs() < 1e-14 {
                // swap with largest
                let mut maxi = k;
                let mut maxp = piv.abs();
                for j in (k+1)..N {
                    let pj = self[[k, j]].abs();
                    if pj > maxp {
                        maxi = j;
                        maxp = pj;
                    }
                }
                if maxi == k {
                    return Err(Error::SingularMatrix);
                }
                // swap rows
                self.rows.swap(k, maxi);
                p.swap(k, maxi);
                pdet *= -1.0;
            }
            let piv = self[[k, k]];

            for i in (k+1)..N {
                let f = - self[[i, k]] / piv;

                self.rows[i] += self.rows[k] * f;
            }
        }

        let mut det = pdet;
        for k in 0..N {
            det *= self[[k, k]];
        }

        Ok(det)
    }



    pub fn plu(mut self) -> Result<([usize; N], Self, Self), Error> {
        let mut l = Self::eye();

        let mut p: [usize; N] = [0; N];
        for i in 0..N {
            p[i] = i;
        }


        for k in 0..(N-1) {
            let piv = self[[k, k]];

            if piv.abs() < 1e-14 {
                // swap with largest
                let mut maxi = k;
                let mut maxp = piv.abs();
                for j in (k+1)..N {
                    let pj = self[[k, j]].abs();
                    if pj > maxp {
                        maxi = j;
                        maxp = pj;
                    }
                }
                if maxi == k {
                    return Err(Error::SingularMatrix);
                }
                // swap rows
                self.rows.swap(k, maxi);
                l.rows.swap(k, maxi);
                p.swap(k, maxi);
            }
            let piv = self[[k, k]];

            for i in (k+1)..N {
                let f = - self[[i, k]] / piv;

                self.rows[i] += self.rows[k] * f;
                l[[i, k]] = -f;
                //l.rows[i] += l.rows[k] * f;
            }
        }

        Ok((p, l, self))
    }

}



// non square matrix decompositions
impl<const M: usize, const N: usize> Matrix<M, N> {


    pub fn qr(self) -> Result<(Matrix<M, M>, Matrix<M, N>), Error> {

        let mut q = Matrix::<M, M>::new();

        // compute the q matrix
        for i in 0..M {

            let mut ui = self.col(i);
            let ai = ui;

            for j in 0..i {
                let uj = q.col(j);
                let uj_dot_uj = uj.dot(uj);
                if uj_dot_uj.abs() < 1e-14 {
                    return Err(Error::SingularMatrix);
                }
                ui -= uj.dot(ai)/uj_dot_uj * uj;
            }

            q.set_col(i, ui);
        }
        // normalize the columns
        for i in 0..M {
            let qi = q.col(i);
            q.set_col(i, qi / qi.norm());
        }

        let mut r = q.transpose() * self;

        for i in 0..M {
            for j in 0..i {
                r[[i, j]] = 0.0;
            }
        }

        Ok((q, r))
    }


    pub fn pseudo_inverse(self) -> Result<Matrix<N, M>, Error> {
        let a_t = self.transpose();
        let at_a = a_t * self;

        Ok(at_a.inv()? * a_t)
    }


}


impl<const M: usize, const N: usize> Default for Matrix<M, N> {
    fn default() -> Self {
        Self::new()
    }
}

unsafe impl<const M: usize, const N: usize> Equivalence for Matrix<M, N> {
    type Out = mpi::datatype::UserDatatype;
    fn equivalent_datatype() -> Self::Out {
        mpi::datatype::UserDatatype::contiguous((M*N) as i32, &f64::equivalent_datatype())
    }
}


impl<const M: usize, const N: usize> FloatBuffered for Matrix<M, N> {
    fn f64_buffer_size() -> usize {
        M*N
    }
    fn put_in_f64_buffer(&self, buffer: &mut [f64]) {
        assert!(buffer.len() >= N);
        for i in 0..M {
            for j in 0..N {
                buffer[i*N+j] = self[[i, j]];
            }
        }
    }
    fn build_from_f64_buffer(buffer: &[f64]) -> Self {
        assert!(buffer.len() >= N);
        let mut out = Self::new();
        for i in 0..M {
            for j in 0..N {
                out[[i, j]] = buffer[i*N+j];
            }
        }
        out
    }
}
