//
// With these settings, it can reach out ~ 99.45% accuracy at the epoch 7 ~ 20
// Currently this is the best result I can achieve with this library.
//
// Epoch 001: time = 14.219s, Validation Accuracy: 98.53%
// Epoch 002: time = 14.306s, Validation Accuracy: 98.67%
// Epoch 003: time = 13.993s, Validation Accuracy: 99.10%
// Epoch 004: time = 14.438s, Validation Accuracy: 99.24%
// Epoch 005: time = 13.986s, Validation Accuracy: 99.02%
// Epoch 006: time = 14.065s, Validation Accuracy: 99.31%
// Epoch 007: time = 14.241s, Validation Accuracy: 99.45%
//

use simple_conv_network::{load_mnist, ActivationFn};
use simple_conv_network::network::*;
use simple_conv_network::conv_pool_layer::*;
use simple_conv_network::fully_connected_layer::*;
use simple_conv_network::softmax_layer::SoftmaxLayer;

fn main() {
    let mnist_data = load_mnist().expect("Failed to load MNIST dataset");

    let cpl1 = ConvPoolLayerConfig {
        input: (1, 28, 28),
        kernel_size: (5, 5),
        num_filters: 20,
        stride: 1,
        padding: 0,
        pool_size: (2, 2),
        pool_stride: 2,
    };

    let cpl2 = ConvPoolLayerConfig {
        input: (20, 12, 12),
        kernel_size: (5, 5),
        num_filters: 40,
        stride: 1,
        padding: 0,
        pool_size: (2, 2),
        pool_stride: 2,
    };

    let mut network = Network::new(
        vec![
            Box::new(ConvPoolLayer::new(&cpl1)),
            Box::new(ConvPoolLayer::new(&cpl2)),
            Box::new(FullyConnectedLayer::with_activation(40 * 4 * 4, 100, ActivationFn::ReLU)),
            Box::new(SoftmaxLayer::new(100, 10)),
        ],
        NetOptions {
            max_epochs: 20,
            mini_batch_size: 20,
            eta: 0.03,
            regularization_l2: 0.1,
            augment_enable: true,
            augment_multiplier: 2,
            ..NetOptions::default()
        },
    );

    // My M2 Air machine doesn't have any fan, so setting a pause duration
    // to avoid overheating the machine.
    network.set_pause_duration(3.0);

    println!("===============================");
    network.show_me();
    println!("===============================");
    println!("Training...");

    network.sdg(&mnist_data);
}