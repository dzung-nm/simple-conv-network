use mnist_conv_rust::load_mnist::{MnistData, load_mnist};
use mnist_conv_rust::network::*;

fn main() {
    let mnist_data = load_mnist().expect("Failed to load MNIST dataset");

    let size = 10000; // Test with a smaller subset of the data for faster training
    let data = MnistData {
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
        sizes: vec![784, 30, 10],
        cost_function: CostFunctions::CrossEntropy,
        weight_init_method: WeightInitMethods::Xavier,
        max_epochs: 10,
        mini_batch_size: 10,
        eta: 0.5,
        regularization_l1: None,
        regularization_l2: Some(1.0),
        stop_early: true,
        stop_early_patience: 20,
        stop_early_min_delta: 0.1,
    };

    let mut network = Network::new(network_options);
    println!("{}", network.options);

    network.sdg(&data);
}
