use ndarray::Array2;

use crate::base_layer::*;
use crate::base_net::*;
use crate::sigmoid::sigmoid_prime;
use crate::types::*;

#[derive(Debug)]
pub enum CostFunctions {
    Quadratic,
    CrossEntropy,
}

pub struct CnnNet {
    base: BaseNet,
    pub layers: Vec<Box<dyn Layer>>,

    // cost_function will depend on the type of the last layer
    cost_function: CostFunctions,
}

impl CnnNet {
    pub fn new(layers: Vec<Box<dyn Layer>>, options: NetOptions) -> Self {
        if layers.len() < 2 {
            panic!("Network must have at least 2 layers");
        }

        // Validate that the output size of each layer matches the input size of the next layer
        for i in 1..layers.len() {
            if layers[i - 1].get_base().output_size != layers[i].get_base().input_size {
                println!(
                    "i = {}, layers[i-1].get_base().output_size = {}, layers[i].get_base().input_size = {}",
                    i,
                    layers[i - 1].get_base().output_size,
                    layers[i].get_base().input_size
                );
                panic!("Input and output layers do not match");
            }
        }

        // If a layer is a Softmax layer, it should be the final layer
        for i in 0..layers.len() - 1 {
            if layers[i].get_type() == LayerTypes::Softmax {
                panic!("Softmax layer must be the final layer in the network");
            }
        }

        let cost_function = match layers.last().unwrap().get_type() {
            LayerTypes::Softmax => CostFunctions::CrossEntropy,
            _ => CostFunctions::Quadratic,
        };

        CnnNet {
            base: BaseNet::new(options),
            layers,
            cost_function,
        }
    }

    pub fn show_me(&self) {
        println!("CnnNet with {}", self.base.options.display());
        self.layers.iter().for_each(|layer| layer.show_me());
    }
}

impl FeedForwardNet for CnnNet {
    fn get_base(&self) -> &BaseNet {
        &self.base
    }

    fn get_base_mut(&mut self) -> &mut BaseNet {
        &mut self.base
    }

    fn feed_forward(&self, x: &Array2<f64>) -> Array2<f64> {
        let mut activation = x.clone();
        for layer in &self.layers {
            let data = layer.forward(&activation);
            activation = data.activation;
        }
        activation
    }

    fn back_propagate(&mut self, x: &Array2<f64>, y: &Array2<f64>) {
        let n = self.layers.len();

        // feedforward
        // Collect LayerData (z, activation) for every layer so that the backward
        // pass can use a_{l-1} as the input to layer l.
        let mut forward_data: Vec<LayerData> = Vec::with_capacity(n);
        for i in 0..n {
            let input = if i == 0 {
                x
            } else {
                &forward_data[i - 1].activation
            };
            let data = self.layers[i].forward(input);
            forward_data.push(data);
        }

        // Calculate output error δ for the final layer based on the cost function
        let mut output_error = match self.cost_function {
            CostFunctions::CrossEntropy => {
                // For cross-entropy cost with softmax, delta is the difference activation - target
                // This is simplified because softmax derivative cancels out with cross-entropy gradient
                &forward_data[n - 1].activation - y
            }
            CostFunctions::Quadratic => {
                // delta = (output - y) * sigmoidPrime(z)
                (&forward_data[n - 1].activation - y) * sigmoid_prime(&forward_data[n - 1].z)
            }
        };

        // backward pass: compute δ for each layer and accumulate ∇W and ∇b
        for l in (0..n).rev() {
            let input = if l == 0 {
                x
            } else {
                &forward_data[l - 1].activation
            };
            let data = self.layers[l].backward(input, &output_error);
            // data.activation = W_l^T · δ_l  →  becomes the error signal for layer l-1
            output_error = data.activation;
        }
    }

    fn update_mini_batch(&mut self, mini_batch: Vec<&TrainingItem>, training_data_size: usize) {
        let base_net = &self.get_base();
        let eta = base_net.options.eta;
        let r_l1 = base_net.options.regularization_l1;
        let r_l2 = base_net.options.regularization_l2;
        let batch_size = mini_batch.len() as f64;
        let data_size = training_data_size as f64;

        // Reset gradients for all layers before processing the mini-batch
        for layer in self.layers.iter_mut() {
            layer.get_base_mut().reset_gradients();
        }

        mini_batch.iter().for_each(|&item| {
            self.back_propagate(&item.0, &item.1);
        });

        // Apply gradient updates
        let scale = eta / batch_size;
        for layer in self.layers.iter_mut() {
            let base = layer.get_base_mut();

            // Skip parameter-free layers (e.g., MaxPoolLayer)
            if base.weights.is_empty() {
                continue;
            }

            // Bias update: b ← b − (η/m) · ∇b
            let db = scale * &base.nabla_b;
            base.biases -= &db;

            // Regularization applied to weights before the gradient step
            if r_l1.is_some() && r_l2.is_some() {
                // Apply both L1 and L2 regularization
                let weight_decay = 1.0 - (eta * r_l2.unwrap()) / data_size;
                base.weights.map_inplace(|w| {
                    *w = *w * weight_decay - eta * r_l1.unwrap() * w.signum() / data_size;
                });
            } else if let Some(l2) = r_l2 {
                // Apply L2 regularization only
                let weight_decay = 1.0 - (eta * l2) / data_size;
                base.weights.map_inplace(|w| *w *= weight_decay);
            } else if let Some(l1) = r_l1 {
                // Apply L1 regularization only
                base.weights.map_inplace(|w| {
                    *w -= eta * l1 * w.signum() / data_size;
                });
            }

            // Weight update: W ← W − (η/m) · ∇W
            let dw = scale * &base.nabla_w;
            base.weights -= &dw;
        }
    }
}
