use crate::core::matrix::DynamicMatrix;



pub trait OdeProblem {
    
    fn size(&self) -> usize;

    fn dudt(&self, dudt: &mut [f64]);

    fn dudt_jacobian(&self, jacobian: &mut DynamicMatrix);

    fn assemble(&self, jacobian: &mut DynamicMatrix, residual: &mut Vec<f64>, state: &Vec<f64>, state_l: &Vec<f64>, state_ll: &Vec<f64>, dt: f64, dt_l: f64) {
        let size = residual.len();

        // (3*u_np1 - 4*u_l + u_ll)/(2*dt) = (du/dt)_np1 

        // 3/2 * I * u  - 4/2*u_l + u_ll/2 = Jac * u + (du/dt - Jac * u)

        // 3/2 * I * u = Jac * u + (du/dt - Jac * u) + 4/2*u_l - 1/2*u_ll

        // (3/2 * I - Jac) * u = (du/dt - Jac * u) + 4/2*u_l - 1/2*u_ll

        // (3/2 * I - Jac) * u = (du/dt - Jac * u) + 4/2*u_l - 1/2*u_ll

        // in assemble, we only fill the - Jac and the (dudt - Jac * u) parts

        self.dudt(residual);
        self.dudt_jacobian(jacobian);

        for i in 0..size {
            for j in 0..size {
                jacobian[[i, j]] = - jacobian[[i, j]];
                if i == j {
                    jacobian[[i, j]] += 3.0 / (2.0 * dt);
                }
            }

            let dudti = residual[i];
            residual[i] = (3.0/2.0*state[i] - 2.0*state_l[i] + 1.0/2.0*state_ll[i]) - dudti;
        }
    }
}



pub struct OdeSolver<T: OdeProblem> {
    problem: T,
    initial_step_size: f64,

    state: Vec<f64>,

    state_l: Vec<f64>,
    state_ll: Vec<f64>,

    tolerance: f64,
}


impl<T: OdeProblem> OdeSolver<T> {
    
    pub fn build(problem: T) -> Self {
        Self { 
            state: vec![0.0; problem.size()],
            state_l: vec![0.0; problem.size()],
            state_ll: vec![0.0; problem.size()],
            problem, 
            initial_step_size: 1e-8 ,
            tolerance: 1e-8,
        }
    }

    pub fn set_initial_step_size(&mut self, step: f64) {
        self.initial_step_size = step;
    }


    pub fn get_state(&self) -> &[f64] {
        &self.state
    }

    pub fn advance_to(&mut self, end_time: f64) -> Result<(), crate::error::Error> {
        let n = self.problem.size();

        // allocate the jacobian
        let mut jac = DynamicMatrix::new(n, n);
        let mut res = vec![0.0; n];

        let mut jacinv = DynamicMatrix::new(n, n);
        let mut update = vec![0.0; n];

        let mut time = 0.0;
        let mut dt = self.initial_step_size;

        let mut dt_l = dt;

        loop {

            // newton loop to solve for time step update
            loop {

                self.problem.assemble(&mut jac, &mut res, &self.state, &self.state_l, &self.state_ll, dt, dt_l);
                let mut error = 0.0;
                for i in 0..n {
                    error += res[i].powi(2);
                }
                error = error.sqrt();
                if error <= self.tolerance {break;}

                jac.inv(&mut jacinv)?;

                jacinv.imul(&mut update, &res);

                for i in 0..n {
                    self.state[i] -= update[i];
                }
            }

            for i in 0..n {
                self.state_ll[i] = self.state_l[i];
                self.state_l[i] = self.state[i];
            }
            dt_l = dt;

            if (time - end_time) > -1e-15 {
                break;
            }

            if (time + dt - end_time) > -1e-15 {
                dt = end_time - time;
            }
            time += dt;
        }

        Ok(())
    }

}








