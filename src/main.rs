use mnist_conv_rust::network::*;
use mnist_conv_rust::load_mnist::{load_mnist, MnistData};

fn main() {
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

    let data = load_mnist().expect("Failed to load MNIST dataset");

    let max = 1000;
    let test_data = MnistData {
        training: data.training.into_iter().take(max).collect(),
        test: data.test.into_iter().take(max).collect(),
        validation: data.validation.into_iter().take(max).collect(),
    };

    network.sdg(&test_data);
}
