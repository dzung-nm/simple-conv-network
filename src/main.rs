mod unzip;
mod load_mnist;

use mnist_conv_rust::types::Dataset;
use mnist_conv_rust::network::*;

use crate::load_mnist::load_mnist;

fn main() {
    let mnist_data = load_mnist().expect("Failed to load MNIST dataset");

    let size = 50000; // Test with a smaller subset of the data for faster training
    let data = Dataset {
        training: mnist_data.training.into_iter().take(size).collect(),
        test: mnist_data.test,
        validation: mnist_data.validation,
    };

    println!(
        "Training data size: {} samples, {} validation samples, {} test samples",
        data.training.len(),
        data.test.len(),
        data.validation.len()
    );

    let network_options = NetworkOptions {
        layers: vec![
            Box::new(FullyConnectedLayer::new(784, 100, WeightInitMethods::Xavier)),
            Box::new(SoftmaxLayer::new(100, 10, WeightInitMethods::Xavier)),
        ],
        max_epochs: 100,
        mini_batch_size: 10,
        eta: 0.1,
        regularization_l1: None,
        regularization_l2: Some(5.0),
        stop_early: true,
        stop_early_patience: 20,
        stop_early_min_delta: 0.1,
    };

    let mut network = Network::new(network_options);
    println!("{}", network.options);

    network.sdg(&data);
}
