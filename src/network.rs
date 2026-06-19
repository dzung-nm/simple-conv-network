use ndarray::Array2;
use rand::prelude::SliceRandom;
use rayon::prelude::*;
use std::time::Instant;

use crate::base_layer::*;
use crate::types::{Dataset, TestItem, TrainingItem};
use crate::utils::{arr_max, get_predicted_label, slice_max};

pub struct NetOptions {
    pub max_epochs: usize,
    pub mini_batch_size: usize,
    pub eta: f64,
    pub regularization_l1: f64,
    pub regularization_l2: f64,
    pub stop_early: bool,
    pub stop_early_patience: usize,
    pub stop_early_min_delta: f64,
}

impl NetOptions {
    pub(crate) fn display(&self) -> String {
        format!(
            "{{ max_epochs: {}, mini_batch_size: {}, eta: {}, regularization_l1: {:?}, \
            regularization_l2: {:?}, stop_early: {}, stop_early_patience: {}, stop_early_min_delta: {} }}",
            self.max_epochs,
            self.mini_batch_size,
            self.eta,
            self.regularization_l1,
            self.regularization_l2,
            self.stop_early,
            self.stop_early_patience,
            self.stop_early_min_delta
        )
    }
}

impl Default for NetOptions {
    fn default() -> NetOptions {
        NetOptions {
            max_epochs: 100,
            mini_batch_size: 10,
            eta: 0.1,
            regularization_l1: 0.0,
            regularization_l2: 0.0,
            stop_early: true,
            stop_early_patience: 20,
            stop_early_min_delta: 0.1,
        }
    }
}

#[derive(PartialEq)]
pub enum LogType {
    Minimal,
    Detailed,
}

pub struct Network {
    pub layers: Vec<Box<dyn Layer>>,
    pub options: NetOptions,

    // For tracking accuracy over epochs
    training_accuracies: Vec<f64>,
    validation_accuracies: Vec<f64>,
    test_accuracies: Vec<f64>,

    // Logging level (minimal or detailed)
    log_type: LogType,
}

impl Network {
    pub fn new(layers: Vec<Box<dyn Layer>>, options: NetOptions) -> Self {
        if layers.len() < 2 {
            panic!("Network must have at least 2 layers");
        }

        // Validate that the output size of each layer matches the input size of the next layer
        for i in 1..layers.len() {
            if layers[i - 1].get_base().output_size != layers[i].get_base().input_size {
                println!(
                    "Layer {} ({}) output size {} does not match Layer {} ({}) input size {}",
                    i - 1,
                    layers[i - 1].get_name(),
                    layers[i - 1].get_base().output_size,
                    i,
                    layers[i].get_name(),
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

        let last_layer = layers.last().unwrap();

        // the last layer should not support dropout, as it is the final layer
        if last_layer.support_dropout() {
            panic!("The last layer should not support dropout");
        }

        Self {
            layers,
            options,
            training_accuracies: Vec::new(),
            validation_accuracies: Vec::new(),
            test_accuracies: Vec::new(),
            log_type: LogType::Minimal,
        }
    }

    pub fn log_more(&mut self) {
        self.log_type = LogType::Detailed;
    }

    pub fn show_me(&self) {
        println!("Network with {}", self.options.display());
        if self.log_type == LogType::Minimal {
            self.layers.iter().for_each(|l| {
                println!("Layer: {}", l.get_name());
            });
        } else {
            self.layers.iter().for_each(|layer| layer.show_me());
        }
    }

    fn feed_forward(&self, x: &Array2<f64>) -> Array2<f64> {
        let mut activation = x.clone();
        for layer in &self.layers {
            let data = layer.forward(&activation, false);
            activation = data.activation;
        }
        activation
    }

    /// Returns a vector of (nabla_w, nabla_b) of size equal to the number of layers,
    /// where each element contains the gradients for that layer.
    fn back_propagate(&self, x: &Array2<f64>, y: &Array2<f64>) -> Vec<(Array2<f64>, Array2<f64>)> {
        let n = self.layers.len();

        // Forward pass: collect ForwardData (z, activation) for every layer
        // so that backward pass can use cached z and a_{l-1}
        let mut forward_data: Vec<ForwardData> = Vec::with_capacity(n);
        for i in 0..n {
            let input = if i == 0 {
                x
            } else {
                &forward_data[i - 1].activation
            };
            let data = self.layers[i].forward(input, true);
            forward_data.push(data);
        }

        let final_layer = &self.layers[n - 1];

        // Calculate output error for the final layer based on the layer type
        let mut output_error = if final_layer.get_type() == LayerTypes::Softmax {
            // Cross-entropy loss with softmax: derivative simplifies to (a_L - y)
            &forward_data[n - 1].activation - y
        } else {
            // Quadratic loss: derivative is (a_L - y) * activation_prime(z_L)
            (&forward_data[n - 1].activation - y)
                * final_layer.activate_prime(&forward_data[n - 1].z)
        };

        let mut results = Vec::with_capacity(n);
        for l in (0..n).rev() {
            let input = if l == 0 {
                x
            } else {
                &forward_data[l - 1].activation
            };
            let backward_data = self.layers[l].backward(input, &output_error, &forward_data[l]);
            // backward_data.input_gradient = W_l^T · δ_l  →  becomes the error signal for layer l-1
            output_error = backward_data.input_gradient;
            results.push((backward_data.nabla_w, backward_data.nabla_b));
        }

        results.reverse();
        results
    }

    fn update_mini_batch(&mut self, mini_batch: Vec<&TrainingItem>, training_data_size: usize) {
        let eta = self.options.eta;
        let r_l1 = self.options.regularization_l1;
        let r_l2 = self.options.regularization_l2;

        let batch_size = mini_batch.len() as f64;
        let data_size = training_data_size as f64;

        // parallelize backpropagation for each item in the mini-batch
        let gradients: Vec<_> = mini_batch
            .par_iter()
            .map(|item| self.back_propagate(&item.0, &item.1))
            .collect();

        // Apply gradient updates
        let scale = eta / batch_size;
        for i in 0..self.layers.len() {
            let layer = self.layers[i].get_base();

            // Skip parameter-free layers (e.g., MaxPoolLayer)
            if layer.weights.is_empty() {
                continue;
            }

            let mut sum_nabla_w = Array2::<f64>::zeros(layer.weights.dim());
            let mut sum_nabla_b = Array2::<f64>::zeros(layer.biases.dim());

            for grad in &gradients {
                sum_nabla_w += &grad[i].0;
                sum_nabla_b += &grad[i].1;
            }

            let mut_layer = self.layers[i].get_base_mut();

            // Bias update: b ← b − (η/m) · ∇b
            let db = scale * &sum_nabla_b;
            mut_layer.biases -= &db;

            // Regularization applied to weights before the gradient step
            if r_l1 > 0.0 && r_l2 > 0.0 {
                // Apply both L1 and L2 regularization
                let weight_decay = 1.0 - (eta * r_l2) / data_size;
                let l1_step = (eta * r_l1) / data_size;
                mut_layer.weights.map_inplace(|w| {
                    *w = *w * weight_decay - l1_step * w.signum();
                });
            } else if r_l2 > 0.0 {
                // Apply L2 regularization only
                let weight_decay = 1.0 - (eta * r_l2) / data_size;
                mut_layer.weights.map_inplace(|w| *w *= weight_decay);
            } else if r_l1 > 0.0 {
                // Apply L1 regularization only
                let l1_step = (eta * r_l1) / data_size;
                mut_layer.weights.map_inplace(|w| {
                    *w -= l1_step * w.signum();
                });
            }

            // Weight update: W ← W − (η/m) · ∇W
            let dw = scale * &sum_nabla_w;
            mut_layer.weights -= &dw;
        }
    }

    fn evaluate_on_training_data(&self, training_data: &Vec<TrainingItem>) -> usize {
        training_data
            .par_iter()
            .map(|item| {
                let output = self.feed_forward(&item.0);
                let predicted = get_predicted_label(&output);
                let actual = item.1.iter().position(|&v| v == 1.0).unwrap();
                predicted == actual
            })
            .filter(|&ok| ok)
            .count()
    }

    fn evaluate_on_test_data(&self, test_data: &Vec<TestItem>) -> usize {
        test_data
            .par_iter()
            .map(|item| {
                let output = self.feed_forward(&item.0);
                get_predicted_label(&output) == item.1 as usize
            })
            .filter(|&ok| ok)
            .count()
    }

    fn log(&mut self, epoch: usize, time_taken: f64, data: &Dataset) {
        let validation_data = &data.validation;
        let validation_accuracy = self.evaluate_on_test_data(validation_data) as f64
            / validation_data.len() as f64
            * 100.0;

        let is_new_record = self.validation_accuracies.is_empty()
            || validation_accuracy > arr_max(&self.validation_accuracies);

        self.validation_accuracies.push(validation_accuracy);

        let validation_label = if is_new_record {
            format!(
                "\x1b[92m\x1b[1mValidation Accuracy: {:.2}%\x1b[0m",
                validation_accuracy
            )
        } else {
            format!("Validation Accuracy: {:.2}%", validation_accuracy)
        };

        if self.log_type == LogType::Detailed {
            let training_data = &data.training;
            let test_data = &data.test;

            let training_accuracy = self.evaluate_on_training_data(training_data) as f64
                / training_data.len() as f64
                * 100.0;
            let test_accuracy =
                self.evaluate_on_test_data(test_data) as f64 / test_data.len() as f64 * 100.0;

            self.training_accuracies.push(training_accuracy);
            self.test_accuracies.push(test_accuracy);

            println!(
                "Epoch {:03}: time = {:.3}s, Training Accuracy: {:.2}%, {}, \x1b[90m\
                Test Accuracy: {:.2}%\x1b[0m",
                epoch + 1,
                time_taken,
                training_accuracy,
                validation_label,
                test_accuracy
            );
        } else {
            println!(
                "Epoch {:03}: time = {:.3}s, {}",
                epoch + 1,
                time_taken,
                validation_label
            );
        }
    }

    fn should_stop_early(&self, accuracies: &Vec<f64>, patience: usize, min_delta: f64) -> bool {
        if accuracies.len() <= patience {
            return false;
        }

        let recent_max = slice_max(&accuracies[accuracies.len() - patience..]);
        let previous_max = slice_max(&accuracies[..accuracies.len() - patience]);

        if recent_max < previous_max + min_delta {
            println!(
                "Early stopping triggered: recent max {:.2}% is not greater than previous \
                max {:.2}% by at least {}",
                recent_max, previous_max, min_delta
            );
            return true;
        }

        false
    }

    pub fn sdg(&mut self, data: &Dataset) {
        let options = &self.options;
        let max_epochs = options.max_epochs;
        let mini_batch_size = options.mini_batch_size;
        let stop_early = options.stop_early;
        let stop_early_patience = options.stop_early_patience;
        let stop_early_min_delta = options.stop_early_min_delta;

        let training_data = &data.training;
        let training_data_size = training_data.len();

        self.training_accuracies.clear();
        self.validation_accuracies.clear();
        self.test_accuracies.clear();

        let mut indices: Vec<usize> = (0..training_data_size).collect();

        for epoch in 0..max_epochs {
            let start = Instant::now();

            indices.shuffle(&mut rand::rng());
            indices.chunks(mini_batch_size).for_each(|batch_indices| {
                let mini_batch = batch_indices
                    .iter()
                    .map(|&i| &training_data[i])
                    .collect::<Vec<_>>();
                self.update_mini_batch(mini_batch, training_data_size);
            });

            let time_taken = start.elapsed();
            self.log(epoch, time_taken.as_secs_f64(), data);

            if stop_early
                && self.should_stop_early(
                    &self.validation_accuracies,
                    stop_early_patience,
                    stop_early_min_delta,
                )
            {
                break;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::fully_connected_layer::FullyConnectedLayer;
    use crate::softmax_layer::SoftmaxLayer;

    #[test]
    #[should_panic = "Network must have at least 2 layers"]
    fn test_at_least_two_layers() {
        let layers: Vec<Box<dyn Layer>> = vec![Box::new(FullyConnectedLayer::new(100, 10))];
        Network::new(layers, NetOptions::default());
    }

    #[test]
    #[should_panic = "Input and output layers do not match"]
    fn test_input_output_not_match() {
        let layers: Vec<Box<dyn Layer>> = vec![
            Box::new(FullyConnectedLayer::new(784, 100)),
            Box::new(FullyConnectedLayer::new(50, 10)),
        ];
        Network::new(layers, NetOptions::default());
    }

    #[test]
    #[should_panic = "Softmax layer must be the final layer in the network"]
    fn test_softmax_not_match() {
        let layers: Vec<Box<dyn Layer>> = vec![
            Box::new(SoftmaxLayer::new(784, 100)),
            Box::new(FullyConnectedLayer::new(100, 10)),
        ];
        Network::new(layers, NetOptions::default());
    }

    #[test]
    #[should_panic = "The last layer should not support dropout"]
    fn test_dropout_not_match() {
        let layers: Vec<Box<dyn Layer>> = vec![
            Box::new(FullyConnectedLayer::new(784, 30)),
            Box::new(FullyConnectedLayer::with_dropout(
                30,
                10,
                ActivationFn::ReLU,
                0.5,
            )),
        ];
        Network::new(layers, NetOptions::default());
    }
}
