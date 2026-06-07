use ndarray::Array2;
use rand::seq::SliceRandom;
use std::time::Instant;

use crate::box_muller::box_muller_random;
use crate::sigmoid::{sigmoid, sigmoid_prime};
use crate::types::{Dataset, TestItem, TrainingItem};
use crate::utils::arr_max;

#[derive(Debug)]
pub enum CostFunctions {
    Quadratic,
    CrossEntropy,
}

#[derive(Debug)]
pub enum WeightInitMethods {
    Standard,
    Xavier,
    He,
}

pub struct BaseLayer {
    input_size: usize,
    output_size: usize,
    weight_init_method: WeightInitMethods,
    weights: Array2<f64>,
    biases: Array2<f64>,
}

impl BaseLayer {
    pub fn new(n_in: usize, n_out: usize, wim: WeightInitMethods) -> BaseLayer {
        let weights = match wim {
            WeightInitMethods::Standard => {
                Array2::from_shape_fn((n_out, n_in), |_| box_muller_random())
            }
            WeightInitMethods::Xavier => Array2::from_shape_fn((n_out, n_in), |_| {
                box_muller_random() * (1.0 / (n_in as f64).sqrt())
            }),
            WeightInitMethods::He => Array2::from_shape_fn((n_out, n_in), |_| {
                box_muller_random() * (2.0 / (n_in as f64)).sqrt()
            }),
        };

        let biases = Array2::from_shape_fn((n_out, 1), |_| box_muller_random());

        Self {
            input_size: n_in,
            output_size: n_out,
            weight_init_method: wim,
            weights,
            biases,
        }
    }
}

pub trait Layer {
    fn get_base(&self) -> &BaseLayer;
    fn get_base_mut(&mut self) -> &mut BaseLayer;
    fn get_name(&self) -> String;
}

pub struct FullyConnectedLayer {
    base: BaseLayer,
}

impl FullyConnectedLayer {
    pub fn new(input_size: usize, output_size: usize, method: WeightInitMethods) -> Self {
        FullyConnectedLayer {
            base: BaseLayer::new(input_size, output_size, method),
        }
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
        "FullyConnectedLayer".to_string()
    }
}

pub struct SoftmaxLayer {
    base: BaseLayer,
}

impl SoftmaxLayer {
    pub fn new(input_size: usize, output_size: usize, method: WeightInitMethods) -> Self {
        SoftmaxLayer {
            base: BaseLayer::new(input_size, output_size, method),
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
        "SoftmaxLayer".to_string()
    }
}

pub struct NetworkOptions {
    pub layers: Vec<Box<dyn Layer>>,
    pub cost_function: CostFunctions,
    pub max_epochs: usize,
    pub mini_batch_size: usize,
    pub eta: f64,
    pub regularization_l1: Option<f64>,
    pub regularization_l2: Option<f64>,
    pub stop_early: bool,
    pub stop_early_patience: usize,
    pub stop_early_min_delta: f64,
}

impl std::fmt::Display for NetworkOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.layers.iter().for_each(|layer| {
            let base_layer = layer.get_base();
            writeln!(
                f,
                "Layer {}: input_size = {}, output_size = {}, weight_init_method = {:?}",
                layer.get_name(),
                base_layer.input_size,
                base_layer.output_size,
                base_layer.weight_init_method
            )
            .unwrap();
        });
        write!(
            f,
            "NetworkOptions {{ cost_function: {:?}, \
            max_epochs: {}, mini_batch_size: {}, eta: {}, regularization_l1: {:?}, \
            regularization_l2: {:?}, stop_early: {}, stop_early_patience: {}, stop_early_min_delta: {} }}",
            self.cost_function,
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

pub struct Network {
    pub options: NetworkOptions,

    // For tracking accuracy over epochs
    training_accuracies: Vec<f64>,
    validation_accuracies: Vec<f64>,
    test_accuracies: Vec<f64>,

    // We use these variables to store the gradients during backpropagation
    // to avoid reallocating them for each training example
    nabla_b: Vec<Array2<f64>>,
    nabla_w: Vec<Array2<f64>>,
}

impl Network {
    pub fn new(options: NetworkOptions) -> Self {
        let layers = &options.layers;

        if layers.len() < 2 {
            panic!("Network must have at least 2 layers");
        }

        // Validate that the output size of each layer matches the input size of the next layer
        for i in 1..layers.len() {
            if layers[i - 1].get_base().output_size != layers[i].get_base().input_size {
                panic!("Input and output layers do not match");
            }
        }

        // If a layer is a Softmax layer, it should be the final layer
        for i in 0..layers.len() - 1 {
            if layers[i].get_name() == "SoftmaxLayer" {
                panic!("Softmax layer must be the final layer in the network");
            }
        }

        let nabla_b = layers
            .iter()
            .map(|layer| Array2::zeros(layer.get_base().biases.dim()))
            .collect::<Vec<_>>();

        let nabla_w = layers
            .iter()
            .map(|layer| Array2::zeros(layer.get_base().weights.dim()))
            .collect::<Vec<_>>();

        Network {
            options,
            training_accuracies: Vec::new(),
            validation_accuracies: Vec::new(),
            test_accuracies: Vec::new(),
            nabla_b,
            nabla_w,
        }
    }

    fn back_propagate(&mut self, x: &Array2<f64>, y: &Array2<f64>) {
        // feedforward
        let mut activations: Vec<Array2<f64>> = Vec::new();
        let mut zs: Vec<Array2<f64>> = Vec::new();
        for i in 0..self.options.layers.len() {
            let layer = &self.options.layers[i].get_base();
            let a = if i == 0 { x } else { &activations[i - 1] };
            // z = w * a + b (using BLAS-accelerated matrix multiplication)
            let z = layer.weights.dot(a) + &layer.biases;
            activations.push(sigmoid(&z));
            zs.push(z);
        }

        // backward pass
        let last_activation = activations.last().unwrap();
        let last_z = zs.last().unwrap();

        let mut delta = match self.options.cost_function {
            CostFunctions::CrossEntropy => {
                // For cross-entropy cost function, the delta is just the output error
                last_activation - y
            }
            CostFunctions::Quadratic => {
                // delta = (output - y) * sigmoidPrime(z)
                (last_activation - y) * sigmoid_prime(&last_z)
            }
        };

        for l in (0..self.options.layers.len()).rev() {
            let layer = &self.options.layers[l].get_base();

            let a_prev = if l == 0 { x } else { &activations[l - 1] };
            self.nabla_b[l] += &delta;
            self.nabla_w[l] += &delta.dot(&a_prev.t());

            if l > 0 {
                let w_transpose = layer.weights.t();
                let z_prev = &zs[l - 1];
                let sp = sigmoid_prime(&z_prev);
                delta = w_transpose.dot(&delta) * sp;
            }
        }
    }

    fn update_mini_batch(&mut self, mini_batch: &Vec<&TrainingItem>, training_data_size: usize) {
        let eta = self.options.eta;
        let r_l1 = self.options.regularization_l1;
        let r_l2 = self.options.regularization_l2;

        self.nabla_b.iter_mut().for_each(|nb| nb.fill(0.0));
        self.nabla_w.iter_mut().for_each(|nw| nw.fill(0.0));

        mini_batch.iter().for_each(|&item| {
            self.back_propagate(&item.0, &item.1);
        });

        for i in 0..self.options.layers.len() {
            let layer = &mut self.options.layers[i].get_base_mut();

            let eta_over_batch_size = eta / mini_batch.len() as f64;
            self.nabla_b[i].map_inplace(|nb| *nb *= eta_over_batch_size);
            self.nabla_w[i].map_inplace(|nw| *nw *= eta_over_batch_size);
            layer.biases -= &self.nabla_b[i];

            let data_size = training_data_size as f64;

            // Apply regularization to weights before updating
            if r_l1.is_some() && r_l2.is_some() {
                // Apply both L1 and L2 regularization
                let weight_decay = 1.0 - (eta * r_l2.unwrap()) / data_size;
                layer.weights.map_inplace(|w| {
                    *w = *w * weight_decay - eta * r_l1.unwrap() * w.signum() / data_size;
                });
            } else if r_l2.is_some() {
                // Apply L2 regularization only
                let weight_decay = 1.0 - (eta * r_l2.unwrap()) / data_size;
                layer.weights.map_inplace(|w| *w *= weight_decay);
            } else if r_l1.is_some() {
                // Apply L1 regularization only
                layer.weights.map_inplace(|w| {
                    *w -= eta * r_l1.unwrap() * w.signum() / data_size;
                });
            }

            layer.weights -= &self.nabla_w[i];
        }
    }

    fn feed_forward(&self, x: &Array2<f64>) -> Array2<f64> {
        let mut activation = x.clone();
        for i in 0..self.options.layers.len() {
            let layer = &self.options.layers[i].get_base();
            let z = layer.weights.dot(&activation) + &layer.biases;
            activation = sigmoid(&z);
        }
        activation
    }

    fn evaluate_on_training_data(&self, training_data: &Vec<TrainingItem>) -> usize {
        training_data
            .iter()
            .map(|item| {
                let output = self.feed_forward(&item.0);
                let data = output.iter().cloned().collect::<Vec<f64>>();
                let predicted = data.iter().position(|&v| v == arr_max(&data)).unwrap();
                let actual = item.1.iter().position(|&v| v == 1.0).unwrap();
                predicted == actual
            })
            .filter(|&is_correct| is_correct)
            .count()
    }

    fn evaluate_on_test_data(&self, test_data: &Vec<TestItem>) -> usize {
        test_data
            .iter()
            .map(|item| {
                let output = self.feed_forward(&item.0);
                let data = output.iter().cloned().collect::<Vec<f64>>();
                let predicted = data.iter().position(|&v| v == arr_max(&data)).unwrap();
                predicted == item.1 as usize
            })
            .filter(|&is_correct| is_correct)
            .count()
    }

    fn calculate_accuracy_and_log(&mut self, epoch: usize, time_taken: f64, data: &Dataset) {
        let training_data = &data.training;
        let validation_data = &data.validation;
        let test_data = &data.test;

        let training_results = self.evaluate_on_training_data(&training_data);
        let training_accuracy = (training_results as f64 / training_data.len() as f64) * 100.0;
        self.training_accuracies.push(training_accuracy);

        let validation_results = self.evaluate_on_test_data(&validation_data);
        let validation_accuracy =
            (validation_results as f64 / validation_data.len() as f64) * 100.0;
        let is_new_validation_record = self.validation_accuracies.len() == 0
            || validation_accuracy > arr_max(&self.validation_accuracies);
        self.validation_accuracies.push(validation_accuracy);

        let test_results = self.evaluate_on_test_data(&test_data);
        let test_accuracy = (test_results as f64 / test_data.len() as f64) * 100.0;
        self.test_accuracies.push(test_accuracy);

        let validation_label = if is_new_validation_record {
            format!(
                "\x1b[92m\x1b[1mValidation Accuracy: {:.2}%\x1b[0m",
                validation_accuracy
            )
        } else {
            format!("Validation Accuracy: {:.2}%", validation_accuracy)
        };
        println!(
            "Epoch {:03}: time = {:.3}s, Training Accuracy: {:.2}%, {}, \x1b[90m\
            Test Accuracy: {:.2}%\x1b[0m",
            epoch + 1,
            time_taken,
            training_accuracy,
            validation_label,
            test_accuracy
        )
    }

    fn should_stop_early(&self, accuracies: &Vec<f64>) -> bool {
        let patience = self.options.stop_early_patience;
        let min_delta = self.options.stop_early_min_delta;

        if accuracies.len() <= patience {
            return false;
        }

        let recent_accuracies = accuracies[accuracies.len() - patience..].to_vec();
        let recent_max = arr_max(&recent_accuracies);
        let previous_max = arr_max(&accuracies[..accuracies.len() - patience].to_vec());

        if recent_max < previous_max + min_delta {
            println!(
                "Early stopping triggered: recent max accuracy {:.2}% is not greater than previous \
                max accuracy {:.2}% by at least {}",
                recent_max, previous_max, min_delta
            );
            return true;
        }

        false
    }

    pub fn sdg(&mut self, data: &Dataset) {
        let max_epochs = self.options.max_epochs;
        let mini_batch_size = self.options.mini_batch_size;
        let stop_early = self.options.stop_early;

        let training_data = &data.training;
        let training_data_size = training_data.len();

        self.training_accuracies.clear();
        self.validation_accuracies.clear();
        self.test_accuracies.clear();

        let mut indices: Vec<usize> = (0..training_data_size).collect();

        for epoch in 0..max_epochs {
            let start = Instant::now();

            indices.shuffle(&mut rand::rng());

            let mini_batches = indices
                .chunks_exact(mini_batch_size)
                .map(|indices_batch| {
                    indices_batch
                        .iter()
                        .map(|&i| &training_data[i])
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();

            mini_batches
                .iter()
                .for_each(|mini_batch| self.update_mini_batch(mini_batch, training_data_size));

            let time_taken = start.elapsed();
            self.calculate_accuracy_and_log(epoch, time_taken.as_secs_f64(), &data);

            if stop_early && self.should_stop_early(&self.validation_accuracies) {
                break;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_network_initialization() {
        let options = NetworkOptions {
            layers: vec![
                Box::new(FullyConnectedLayer::new(
                    784,
                    100,
                    WeightInitMethods::Xavier,
                )),
                Box::new(SoftmaxLayer::new(100, 10, WeightInitMethods::Xavier)),
            ],
            cost_function: CostFunctions::CrossEntropy,
            max_epochs: 10,
            mini_batch_size: 10,
            eta: 0.1,
            regularization_l1: None,
            regularization_l2: Some(5.0),
            stop_early: true,
            stop_early_patience: 20,
            stop_early_min_delta: 0.1,
        };

        let network = Network::new(options);
        assert_eq!(network.options.layers.len(), 2);
        assert_eq!(network.options.max_epochs, 10);
        assert_eq!(network.options.mini_batch_size, 10);
        assert_eq!(network.options.eta, 0.1);
        assert_eq!(network.options.regularization_l1, None);
        assert_eq!(network.options.regularization_l2, Some(5.0));
        assert_eq!(network.options.stop_early, true);
        assert_eq!(network.options.stop_early_patience, 20);
        assert_eq!(network.options.stop_early_min_delta, 0.1);
    }

    #[test]
    #[should_panic = "Network must have at least 2 layers"]
    fn test_at_least_two_layers() {
        let options = NetworkOptions {
            layers: vec![Box::new(SoftmaxLayer::new(
                100,
                10,
                WeightInitMethods::Xavier,
            ))],
            cost_function: CostFunctions::CrossEntropy,
            max_epochs: 10,
            mini_batch_size: 10,
            eta: 0.1,
            regularization_l1: None,
            regularization_l2: Some(5.0),
            stop_early: true,
            stop_early_patience: 20,
            stop_early_min_delta: 0.1,
        };
        Network::new(options);
    }

    #[test]
    #[should_panic = "Input and output layers do not match"]
    fn test_input_output_not_match() {
        let options = NetworkOptions {
            layers: vec![
                Box::new(FullyConnectedLayer::new(
                    784,
                    100,
                    WeightInitMethods::Xavier,
                )),
                Box::new(SoftmaxLayer::new(101, 10, WeightInitMethods::Xavier)),
            ],
            cost_function: CostFunctions::CrossEntropy,
            max_epochs: 10,
            mini_batch_size: 10,
            eta: 0.1,
            regularization_l1: None,
            regularization_l2: Some(5.0),
            stop_early: true,
            stop_early_patience: 20,
            stop_early_min_delta: 0.1,
        };
        Network::new(options);
    }

    #[test]
    #[should_panic = "Softmax layer must be the final layer in the network"]
    fn test_softmax_not_match() {
        let options = NetworkOptions {
            layers: vec![
                Box::new(SoftmaxLayer::new(784, 100, WeightInitMethods::Xavier)),
                Box::new(SoftmaxLayer::new(100, 10, WeightInitMethods::Xavier)),
            ],
            cost_function: CostFunctions::Quadratic,
            max_epochs: 10,
            mini_batch_size: 10,
            eta: 0.1,
            regularization_l1: None,
            regularization_l2: Some(5.0),
            stop_early: true,
            stop_early_patience: 20,
            stop_early_min_delta: 0.1,
        };
        Network::new(options);
    }
}
