use ndarray::Array2;

use crate::base_layer::*;
use crate::box_muller::box_muller_random;
use crate::sigmoid::*;

pub struct SigmoidLayer {
    base: BaseLayer,
    dropout_rate: f64,
}

impl SigmoidLayer {
    pub fn new(n_in: usize, n_out: usize) -> Self {
        let weights = Array2::from_shape_fn((n_out, n_in), |_| {
            box_muller_random() * (1.0 / (n_in as f64).sqrt()) // Xavier initialization
        });
        let biases = Array2::from_shape_fn((n_out, 1), |_| box_muller_random());

        SigmoidLayer {
            base: BaseLayer {
                input_size: n_in,
                output_size: n_out,
                weights,
                biases,
            },
            dropout_rate: 0.0,
        }
    }

    /// Create a new SigmoidLayer with optional dropout
    /// dropout_rate - Probability of dropping a neuron (0.0 = no dropout, 0.5 = 50% dropout)
    pub fn new_with_dropout(n_in: usize, n_out: usize, dropout_rate: f64) -> Self {
        let mut layer = SigmoidLayer::new(n_in, n_out);
        layer.dropout_rate = dropout_rate.clamp(0.0, 1.0);
        layer
    }
}

impl Layer for SigmoidLayer {
    fn get_base(&self) -> &BaseLayer {
        &self.base
    }

    fn get_base_mut(&mut self) -> &mut BaseLayer {
        &mut self.base
    }

    fn get_name(&self) -> String {
        if self.dropout_rate > 0.0 {
            return format!(
                "SigmoidLayer (Xavier init, dropout={:.2})",
                self.dropout_rate
            );
        };

        "SigmoidLayer (Xavier init, no dropout)".to_string()
    }

    fn support_dropout(&self) -> bool {
        self.dropout_rate > 0.0
    }

    fn get_type(&self) -> LayerTypes {
        LayerTypes::Sigmoid
    }

    fn activate(&self, z: &Array2<f64>) -> Array2<f64> {
        sigmoid(z)
    }

    fn activate_prime(&self, z: &Array2<f64>) -> Array2<f64> {
        sigmoid_prime(z)
    }

    fn forward(&self, input: &Array2<f64>, is_training: bool) -> ForwardData {
        let base = self.get_base();
        let z = base.weights.dot(input) + &base.biases;
        let mut activation = self.activate(&z);

        let dropout_mask = if is_training && self.dropout_rate > 0.0 {
            let keep_prob = 1.0 - self.dropout_rate;
            let scale = 1.0 / keep_prob; // Inverted dropout: scale up during training

            // Create dropout mask: 0 for dropped neurons, scale for kept neurons
            let mask = Array2::from_shape_fn(activation.dim(), |_| {
                if rand::random::<f64>() < keep_prob {
                    scale
                } else {
                    0.0
                }
            });

            // Apply mask to activation
            activation = activation * &mask;
            Some(mask)
        } else {
            None // No dropout during inference
        };

        ForwardData {
            z,
            activation,
            cache: None,
            dropout_mask,
        }
    }

    fn backward(
        &self,
        input: &Array2<f64>,
        output_error: &Array2<f64>,
        forward_data: &ForwardData,
    ) -> BackwardData {
        // Apply dropout mask to output_error if it exists
        let masked_error = if let Some(ref mask) = forward_data.dropout_mask {
            output_error * mask
        } else {
            output_error.clone()
        };

        let delta = &masked_error * self.activate_prime(&forward_data.z);
        let nabla_w = delta.dot(&input.t());

        // Propagated error for the previous layer: W_l^T · δ_l
        let input_gradient = self.get_base().weights.t().dot(&delta);

        BackwardData {
            input_gradient,
            nabla_b: delta,
            nabla_w,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_forward_training() {
        let layer = SigmoidLayer::new_with_dropout(4, 3, 0.5);
        let input = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();

        let forward_data = layer.forward(&input, true);

        // Of course, there must be a dropout mask during training
        assert!(forward_data.dropout_mask.is_some());

        let activation = forward_data.activation;
        let dropout_mask = forward_data.dropout_mask.unwrap();

        // There should have at least one neuron dropped (0.0) and at least one active neuron
        let num_dropped = activation.iter().filter(|&&x| x == 0.0).count();
        let num_active = activation.iter().filter(|&&x| x != 0.0).count();
        assert!(num_dropped > 0);
        assert!(num_active > 0);

        // In this case, scale = 1 / (1 - 0.5) = 2.0
        // So, there should have at least one mask value = 0.0 and at least one mask value = 2.0
        let num_mask_dropped = dropout_mask.iter().filter(|&&x| x == 0.0).count();
        let num_active = dropout_mask.iter().filter(|&&x| x != 0.0).count();
        assert!(num_mask_dropped > 0);
        assert!(num_active > 0);
    }

    #[test]
    fn test_forward_inference() {
        let input = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();

        let layer = SigmoidLayer::new(4, 3);
        let forward_data = layer.forward(&input, false);
        assert!(forward_data.dropout_mask.is_none());

        let layer = SigmoidLayer::new_with_dropout(4, 3, 0.0);
        let forward_data = layer.forward(&input, false);
        assert!(forward_data.dropout_mask.is_none());
    }
}
