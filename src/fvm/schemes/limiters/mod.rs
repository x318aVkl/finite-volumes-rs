

pub trait LimiterScheme {
    fn get_limiter(&self, r: f64) -> f64;
}


impl<'b> LimiterScheme for Box<dyn LimiterScheme + 'b> {
    fn get_limiter(&self, r: f64) -> f64 {
        self.as_ref().get_limiter(r)
    }
}


pub struct LimitedLinear(pub f64);

impl LimiterScheme for LimitedLinear {
    fn get_limiter(&self, r: f64) -> f64 {
        (2.0*r*self.0).min(1.0)
    }
}


pub struct MinMod;

impl LimiterScheme for MinMod {
    fn get_limiter(&self, r: f64) -> f64 {
        (1.0*r).min(1.0)
    }
}


pub struct VanLeer;

impl LimiterScheme for VanLeer {
    fn get_limiter(&self, r: f64) -> f64 {
        2.0 * r / (1.0 + r)
    }
}


pub struct VanAlbada;

impl LimiterScheme for VanAlbada {
    fn get_limiter(&self, r: f64) -> f64 {
        (r*r + r) / (r*r + 1.0)
    }
}

pub struct Sweby(pub f64);

impl LimiterScheme for Sweby {
    fn get_limiter(&self, r: f64) -> f64 {
        let beta = self.0;

        let a = r.min(beta);
        let b = (beta*r).min(1.0);

        a.max(b).max(0.0)
    }
}


pub struct SuperBee;

impl LimiterScheme for SuperBee {
    fn get_limiter(&self, r: f64) -> f64 {
        Sweby(2.0).get_limiter(r)
    }
}
