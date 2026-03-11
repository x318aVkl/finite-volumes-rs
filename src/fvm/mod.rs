

//pub mod limiters;
//pub use limiters::compute_limiters;

pub mod traits {


    pub trait Elementwise {
        type Element;

        fn elemwise_min(self, rhs: Self) -> Self;
        fn elemwise_max(self, rhs: Self) -> Self;

        fn elemwise_max_single(self, rhs: Self::Element) -> Self;
        fn elemwise_min_single(self, rhs: Self::Element) -> Self;

        fn elemwise_mul(self, rhs: Self) -> Self;
        fn elemwise_div(self, rhs: Self) -> Self;
        fn elemwise_add(self, rhs: Self) -> Self;
        fn elemwise_sub(self, rhs: Self) -> Self;
        fn elemwise_abs(self) -> Self;
        
        fn elemwise_powf(self, pow: f64) -> Self;

        fn elemwise_map(self, map: impl Fn(Self::Element) -> Self::Element) -> Self;
        fn elemwise_map_zip(self, rhs: Self, map: impl Fn(Self::Element, Self::Element) -> Self::Element) -> Self;
    }


}