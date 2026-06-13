use ndarray::Array2;

use crate::base_layer::*;
use crate::box_muller::box_muller_random;
use crate::softmax::*;

pub struct SoftmaxLayer {
    base: BaseLayer,
}

impl SoftmaxLayer {
    pub fn new(n_in: usize, n_out: usize) -> Self {
        let weights = Array2::from_shape_fn((n_out, n_in), |_| {
            box_muller_random() * (1.0 / (n_in as f64).sqrt()) // Xavier initialization
        });
        let biases = Array2::from_shape_fn((n_out, 1), |_| box_muller_random());
        let nabla_w = Array2::zeros((n_out, n_in));
        let nabla_b = Array2::zeros((n_out, 1));

        SoftmaxLayer {
            base: BaseLayer {
                input_size: n_in,
                output_size: n_out,
                weights,
                biases,
                nabla_w,
                nabla_b,
            },
        }
    }
}

impl Layer for SoftmaxLayer {
    fn get_base(&self) -> &BaseLayer {
        &self.base
    }

    fn get_base_mut(&mut self) -> &mut BaseLayer {
        &mut self.base
    }

    fn get_name(&self) -> String {
        "SoftmaxLayer, weight init method = Xavier".to_string()
    }

    fn get_type(&self) -> LayerTypes {
        LayerTypes::Softmax
    }

    fn activate(&self, z: &Array2<f64>) -> Array2<f64> {
        softmax(z)
    }

    fn activate_prime(&self, z: &Array2<f64>) -> Array2<f64> {
        // Softmax must always be the last layer; activate_prime is never used in the
        // backward chain beyond the last layer.  Returning all-ones means the default
        // backward implementation correctly reduces to:  δ_L = output_error ⊙ 1 = a_L − y.
        Array2::from_shape_fn(z.dim(), |_| 1.0)
    }
}