use crate::base_layer::*;
use crate::box_muller::box_muller_random;
use crate::relu::*;
use crate::sigmoid::*;
use ndarray::Array2;

pub struct FullyConnectedLayer {
    base: BaseLayer,
    dropout_rate: f64,
    activation_fn: ActivationFn,
}

impl FullyConnectedLayer {
    pub fn with_dropout(
        n_in: usize,
        n_out: usize,
        activation_fn: ActivationFn,
        dropout_rate: f64,
    ) -> Self {
        // Use appropriate initialization based on activation function
        let std = match activation_fn {
            ActivationFn::ReLU => (2.0 / n_in as f64).sqrt(), // He initialization for ReLU
            ActivationFn::Sigmoid => (1.0 / n_in as f64).sqrt(), // Xavier initialization for Sigmoid
        };

        let weights = Array2::from_shape_fn((n_out, n_in), |_| box_muller_random() * std);
        let biases = Array2::from_shape_fn((n_out, 1), |_| box_muller_random());

        FullyConnectedLayer {
            base: BaseLayer {
                input_size: n_in,
                output_size: n_out,
                weights,
                biases,
            },
            dropout_rate: dropout_rate.clamp(0.0, 1.0),
            activation_fn,
        }
    }

    pub fn with_activation(n_in: usize, n_out: usize, activation_fn: ActivationFn) -> Self {
        Self::with_dropout(n_in, n_out, activation_fn, 0.0)
    }

    pub fn new(n_in: usize, n_out: usize) -> Self {
        Self::with_dropout(n_in, n_out, ActivationFn::Sigmoid, 0.0)
    }
}

impl Layer for FullyConnectedLayer {
    fn get_base(&self) -> &BaseLayer {
        &self.base
    }

    fn get_base_mut(&mut self) -> &mut BaseLayer {
        &mut self.base
    }

    fn get_name(&self) -> String {
        let dropout_label = if self.dropout_rate > 0.0 {
            format!("dropout={:.2}", self.dropout_rate)
        } else {
            "no dropout".to_string()
        };
        let init_label = match self.activation_fn {
            ActivationFn::ReLU => "He init",
            ActivationFn::Sigmoid => "Xavier init",
        };
        format!(
            "FullyConnectedLayer ({}, {}, activation={:?}, in/out={}/{})",
            init_label,
            dropout_label,
            self.activation_fn,
            self.base.input_size,
            self.base.output_size
        )
    }

    fn clone_layer(&self) -> Box<dyn Layer> {
        Box::new(Self {
            base: self.base.clone(),
            dropout_rate: self.dropout_rate,
            activation_fn: self.activation_fn.clone(),
        })
    }

    fn support_dropout(&self) -> bool {
        self.dropout_rate > 0.0
    }

    fn get_type(&self) -> LayerTypes {
        LayerTypes::FullyConnected
    }

    fn activate(&self, z: &Array2<f64>) -> Array2<f64> {
        match self.activation_fn {
            ActivationFn::Sigmoid => sigmoid(z),
            ActivationFn::ReLU => relu(z),
        }
    }

    fn activate_prime(&self, z: &Array2<f64>) -> Array2<f64> {
        match self.activation_fn {
            ActivationFn::Sigmoid => sigmoid_prime(z),
            ActivationFn::ReLU => relu_prime(z),
        }
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
        // With 20 output neurons and 50% dropout: P(all drop or all active) ≈ 0.00019%
        // That means it is rarely failed.
        let layer = FullyConnectedLayer::with_dropout(4, 20, ActivationFn::Sigmoid, 0.5);
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

        let layer = FullyConnectedLayer::new(4, 3);
        let forward_data = layer.forward(&input, false);
        assert!(forward_data.dropout_mask.is_none());

        let layer = FullyConnectedLayer::with_dropout(4, 3, ActivationFn::ReLU, 0.0);
        let forward_data = layer.forward(&input, false);
        assert!(forward_data.dropout_mask.is_none());
    }
}
