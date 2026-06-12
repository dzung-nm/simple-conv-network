mod unzip;
mod load_mnist;

use mnist_conv_rust::types::Dataset;
use mnist_conv_rust::cnn_net::*;
use mnist_conv_rust::NetOptions;
use mnist_conv_rust::Layer;
use mnist_conv_rust::FeedForwardNet;
use mnist_conv_rust::conv_layer::ConvLayer;
use mnist_conv_rust::max_pool_layer::MaxPoolLayer;
use mnist_conv_rust::sigmoid_layer::SigmoidLayer;
use mnist_conv_rust::softmax_layer::SoftmaxLayer;

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

    let net_options = NetOptions {
        max_epochs: 100,
        mini_batch_size: 10,
        eta: 0.1,
        regularization_l1: None,
        regularization_l2: Some(5.0),
        stop_early: true,
        stop_early_patience: 20,
        stop_early_min_delta: 0.1,
    };

    let layers: Vec<Box<dyn Layer>> = vec![
        Box::new(ConvLayer::new(1, 28, 28, 6, 5, 5, 1, 0)),
        Box::new(MaxPoolLayer::new(6, 24, 24, 2, 2, 2)), // → 6×12×12 = 864
        Box::new(SigmoidLayer::new(864, 30)),
        Box::new(SoftmaxLayer::new(30, 10)),
    ];

    let mut network = CnnNet::new(layers, net_options);

    println!("===============================");
    network.show_me();
    println!("===============================");
    println!("Training...");

    network.sdg(&data);
}
