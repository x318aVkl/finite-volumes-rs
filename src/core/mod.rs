
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


    pub trait FloatBuffered {
        fn f64_buffer_size() -> usize;
        fn put_in_f64_buffer(&self, buffer: &mut [f64]);
        fn build_from_f64_buffer(buffer: &[f64]) -> Self;
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
    }

}


