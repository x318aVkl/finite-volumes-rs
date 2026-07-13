
pub mod mesh;
pub mod sparsity;
pub mod vector;
pub mod matrix;
pub mod error;
pub mod field;
pub mod communicator;
pub mod evaluator;

pub use sparsity::Sparsity;
pub use vector::Vector;
pub use matrix::Matrix;
pub use mesh::Mesh;

pub(crate) use communicator::Communicator;

pub mod traits {

    use super::Matrix;
    use super::Vector;


    pub trait FloatBuffered {
        fn f64_buffer_size() -> usize;
        fn put_in_f64_buffer(&self, buffer: &mut [f64]);
        fn build_from_f64_buffer(buffer: &[f64]) -> Self;
        fn put_single_in_f64_buffer(&mut self, local_id: usize, value: f64);
    }

    impl FloatBuffered for f64 {
        fn f64_buffer_size() -> usize {
            1
        }
        fn put_in_f64_buffer(&self, buffer: &mut [f64]) {
            assert!(buffer.len() > 0);
            buffer[0] = *self;
        }
        fn build_from_f64_buffer(buffer: &[f64]) -> Self {
            assert!(buffer.len() > 0);
            buffer[0]
        }
        fn put_single_in_f64_buffer(&mut self, local_id: usize, value: f64) {
            assert_eq!(local_id, 0);
            *self = value;
        }
    }


    pub trait Unit {
        fn unit() -> Self;
    }

    pub trait Zero {
        fn zero() -> Self;
    }



    impl Unit for f64 {
        fn unit() -> Self {
            1.0
        }
    }

    impl<const N: usize> Unit for Vector<N> {
        fn unit() -> Self {
            Self::one()
        }
    }

    impl<const M: usize, const N: usize> Unit for Matrix<M, N> {
        fn unit() -> Self {
            Self::eye()
        }
    }


    impl Zero for f64 {
        fn zero() -> Self {
            0.0
        }
    }

    impl<const N: usize> Zero for Vector<N> {
        fn zero() -> Self {
            [0.0; N].into()
        }
    }

    impl<const M: usize, const N: usize> Zero for Matrix<M, N> {
        fn zero() -> Self {
            Self::from_f64(0.0)
        }
    }

}


