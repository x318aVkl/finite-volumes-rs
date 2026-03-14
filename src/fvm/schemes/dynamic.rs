use std::ops::{Add, Div, Mul, Neg};

use crate::{Field, Vector, fvm::schemes::{faceinterp::FaceInterpolationScheme, facengrad::FaceNormalGradientScheme, limiters::LimiterScheme, time::TimeScheme}, prelude::{Unit, Zero, geometry}};



pub enum SchemeType {
    FaceInterpolation,
    FaceNormalGradient,
    Limiter,
    Time,
}



pub struct DynamicSchemeSet {

    faceinterp: Option<String>,

    facengrad: Option<String>,

    limiter: Option<String>,

    time: Option<String>,

}


impl Default for DynamicSchemeSet {
    fn default() -> Self {
        Self {
            faceinterp: None,
            facengrad: Some("corrected".to_string()),
            limiter: Some("limited-linear".to_string()),
            time: Some("euler".to_string()),
        }
    }
}



impl DynamicSchemeSet {

    pub fn empty() -> Self {
        Self { faceinterp: None, facengrad: None, limiter: None, time: None }
    }


    pub fn set(&mut self, scheme: SchemeType, value: &str) {
        match scheme {
            SchemeType::FaceInterpolation => self.faceinterp = Some(value.to_string()),
            SchemeType::FaceNormalGradient => self.facengrad = Some(value.to_string()),
            SchemeType::Limiter => self.limiter = Some(value.to_string()),
            SchemeType::Time => self.time = Some(value.to_string()),
        }
    }

    pub fn with(mut self, scheme: SchemeType, value: &str) -> Self {
        self.set(scheme, value);
        self
    }


    pub fn faceinterp<'a, Lhs, Rhs, const DIM: usize>(
        &'a self,
        flux: Option<&'a Field<f64, geometry::Face, DIM>>,
        limiters: Option<&'a Field<f64, geometry::Face, DIM>>,
    ) -> Box<dyn FaceInterpolationScheme<DIM, Lhs=Lhs, Rhs=Rhs> + 'a> 
    where Lhs: Unit + Zero + Mul<f64, Output = Lhs> + 'a,
    Rhs: Unit + Zero + Mul<f64, Output = Rhs> + 'a,
    {
        let scheme = self.faceinterp.as_ref().expect("dynamic scheme set contains a face-interpolation scheme");

        match scheme.as_str() {

            "linear" => {
                Box::new(super::faceinterp::Linear::new())
            },
            "upwind" => {
                let flux = flux.expect("Requested upwind face-interpolation scheme without a face flux field");
                Box::new(super::faceinterp::Upwind::new(flux))
            },
            "limited-linear" => {
                let flux = flux.expect("Requested limited-linear face-interpolation scheme without a face flux field");
                let limiters = limiters.expect("Requested limited-linear face-interpolation scheme without a limiter field");
                Box::new(super::faceinterp::LimitedLinear::new(flux, limiters))
            },

            _ => panic!("Unknown face-interpolation scheme \"{}\"", scheme),
        }
    }

    pub fn facengrad<'a, Lhs, Rhs, G, const DIM: usize>(
        &'a self,
        gradients: Option<&'a Field<G, geometry::Cell, DIM>>,
    ) -> Box<dyn FaceNormalGradientScheme<DIM, Lhs=Lhs, Rhs=Rhs> + 'a> 
        where Lhs: Unit + Zero + Mul<f64, Output = Lhs> + 'a,
    Rhs: Unit + Zero + Mul<f64, Output = Rhs> + Neg<Output = Rhs> + 'a,
    G: Mul<f64, Output = G> + Add<G, Output = G> + Mul<Vector<DIM>, Output=Rhs> + Copy,
    {
        let scheme = self.facengrad.as_ref().expect("dynamic scheme set contains a face-normal-gradient scheme");

        match scheme.as_str() {

            "orthogonal" => {
                Box::new(super::facengrad::Orthogonal::new())
            },

            "corrected" => {
                let gradients = gradients.expect("Requested corrected face-normal-gradient scheme without a gradient field");
                Box::new(super::facengrad::Corrected::new(gradients, 1.0))
            }

            _ => panic!("Unknown face-normal-gradient scheme \"{}\"", scheme)
        }

    }

    pub fn limiter<'a>(&'a self) -> Box<dyn LimiterScheme + 'a> {
        let scheme = self.limiter.as_ref().expect("dynamic scheme set contains a limiter scheme");

        match scheme.as_str() {

            "limited-linear" => {
                Box::new(super::limiters::LimitedLinear(2.0))
            },
            "minmod" => {
                Box::new(super::limiters::MinMod)
            }
            "vanleer" => {
                Box::new(super::limiters::MinMod)
            }
            
            _ => panic!("Unknown limiter scheme \"{}\"", scheme),
        }
    }

    pub fn time<'a, Lhs, Rhs, const DIM: usize>(
        &'a self,
        dt: f64,
        previous: Option<&'a Field<Rhs, geometry::Cell, DIM>>,
    ) -> Box<dyn TimeScheme<DIM, Lhs=Lhs, Rhs=Rhs> + 'a> 
    where 
    Lhs: Unit + Mul<f64, Output = Lhs> + 'a,
    Rhs: Div<f64, Output = Rhs> + Copy,
    {
        let scheme = self.time.as_ref().expect("dynamic scheme set contains a time scheme");

        match scheme.as_str() {

            "euler" => {
                let previous = previous.expect("Requested euler time scheme without a previous value field");
                Box::new(super::time::Euler::new(previous, dt))
            }

            _ => panic!("Unknown time scheme \"{}\"", scheme)
        }
    }

}

