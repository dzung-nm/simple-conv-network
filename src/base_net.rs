// Define a common feedforward neural network

use ndarray::Array2;
use rand::prelude::SliceRandom;
use std::time::Instant;

use crate::types::{Dataset, TestItem, TrainingItem};
use crate::utils::{arr_max, get_predicted_label, slice_max};

pub struct NetOptions {
    pub max_epochs: usize,
    pub mini_batch_size: usize,
    pub eta: f64,
    pub regularization_l1: Option<f64>,
    pub regularization_l2: Option<f64>,
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

pub struct BaseNet {
    pub options: NetOptions,

    // For tracking accuracy over epochs
    training_accuracies: Vec<f64>,
    validation_accuracies: Vec<f64>,
    test_accuracies: Vec<f64>,
}

impl BaseNet {
    pub fn new(options: NetOptions) -> Self {
        Self {
            options,
            training_accuracies: Vec::new(),
            validation_accuracies: Vec::new(),
            test_accuracies: Vec::new(),
        }
    }
}

pub trait FeedForwardNet {
    fn get_base(&self) -> &BaseNet;
    fn get_base_mut(&mut self) -> &mut BaseNet;
    fn feed_forward(&self, x: &Array2<f64>) -> Array2<f64>;
    fn back_propagate(&mut self, x: &Array2<f64>, y: &Array2<f64>);
    fn update_mini_batch(&mut self, mini_batch: Vec<&TrainingItem>, training_data_size: usize);

    fn evaluate_on_training_data(&self, training_data: &Vec<TrainingItem>) -> usize {
        training_data
            .iter()
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
            .iter()
            .map(|item| {
                let output = self.feed_forward(&item.0);
                get_predicted_label(&output) == item.1 as usize
            })
            .filter(|&ok| ok)
            .count()
    }

    fn calculate_accuracy_and_log(&mut self, epoch: usize, time_taken: f64, data: &Dataset) {
        let training_data = &data.training;
        let validation_data = &data.validation;
        let test_data = &data.test;

        let training_accuracy = self.evaluate_on_training_data(training_data) as f64
            / training_data.len() as f64
            * 100.0;
        let validation_accuracy = self.evaluate_on_test_data(validation_data) as f64
            / validation_data.len() as f64
            * 100.0;
        let test_accuracy =
            self.evaluate_on_test_data(test_data) as f64 / test_data.len() as f64 * 100.0;

        let base_net = self.get_base_mut();
        base_net.training_accuracies.push(training_accuracy);
        let is_new_record = base_net.validation_accuracies.is_empty()
            || validation_accuracy > arr_max(&base_net.validation_accuracies);
        base_net.validation_accuracies.push(validation_accuracy);
        base_net.test_accuracies.push(test_accuracy);

        let validation_label = if is_new_record {
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
        );
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

    fn sdg(&mut self, data: &Dataset) {
        let options = &self.get_base().options;
        let max_epochs = options.max_epochs;
        let mini_batch_size = options.mini_batch_size;
        let stop_early = options.stop_early;
        let stop_early_patience = options.stop_early_patience;
        let stop_early_min_delta = options.stop_early_min_delta;

        let training_data = &data.training;
        let training_data_size = training_data.len();

        let base_net = self.get_base_mut();
        base_net.training_accuracies.clear();
        base_net.validation_accuracies.clear();
        base_net.test_accuracies.clear();

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
            self.calculate_accuracy_and_log(epoch, time_taken.as_secs_f64(), data);

            if stop_early
                && self.should_stop_early(
                    &self.get_base().validation_accuracies,
                    stop_early_patience,
                    stop_early_min_delta,
                )
            {
                break;
            }
        }
    }
}
