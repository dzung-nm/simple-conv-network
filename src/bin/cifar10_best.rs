/// This is a neural network implementation for the CIFAR-10 dataset using a simple
/// convolutional network architecture.
///     cargo run --release --bin cifar10_best
/// Currently it can archive 69.10% accuracy with num_samples_per_epoch = 20.000
/// Epoch 015: time = 21.616s, Validation Accuracy: 69.10%

use simple_conv_network::conv_layer::{ConvLayer, ConvLayerConfig};
use simple_conv_network::{load_cifar10, Layer, ActivationFn};
use simple_conv_network::fully_connected_layer::FullyConnectedLayer;
use simple_conv_network::max_pool_layer::*;
use simple_conv_network::network::{NetOptions, Network};
use simple_conv_network::softmax_layer::SoftmaxLayer;

fn main() {
    let data = load_cifar10().expect("Failed to load CIFAR-10 dataset");

    let cv_layer1 = ConvLayerConfig {
        input: (3, 32, 32),
        kernel_size: (5, 5),
        num_filters: 16,
        stride: 1,
        padding: 2,
    };
    let pl_layer1 = PoolLayerConfig {
        input: (16, 32, 32),
        pool_size: (2, 2),
        stride: 2,
    };

    let cv_layer2 = ConvLayerConfig {
        input: (16, 16, 16),
        kernel_size: (5, 5),
        num_filters: 20,
        stride: 1,
        padding: 2,
    };
    let pl_layer2 = PoolLayerConfig {
        input: (20, 16, 16),
        pool_size: (2, 2),
        stride: 2,
    };

    let cv_layer3 = ConvLayerConfig {
        input: (20, 8, 8),
        kernel_size: (5, 5),
        num_filters: 20,
        stride: 1,
        padding: 2,
    };
    let pl_layer3 = PoolLayerConfig {
        input: (20, 8, 8),
        pool_size: (2, 2),
        stride: 2,
    };

    let layers: Vec<Box<dyn Layer>> = vec![
        Box::new(ConvLayer::new(&cv_layer1)), // → 16×32×32
        Box::new(MaxPoolLayer::new(&pl_layer1)), // → 16×16×16
        Box::new(ConvLayer::new(&cv_layer2)), // → 20×16×16
        Box::new(MaxPoolLayer::new(&pl_layer2)), // → 20×8×8
        Box::new(ConvLayer::new(&cv_layer3)), // → 20×8×8
        Box::new(MaxPoolLayer::new(&pl_layer3)), // → 20×4×4
        Box::new(FullyConnectedLayer::with_dropout(20 * 4 * 4, 10, ActivationFn::ReLU, 0.0)), // no dropout
        Box::new(SoftmaxLayer::new(10, 10)),
    ];

    let mut network = Network::new(
        layers,
        NetOptions {
            max_epochs: 50,
            mini_batch_size: 20,
            eta: 0.01,
            regularization_l2: 0.0001,
            augment_enable: true,
            augment_multiplier: 3,
            num_samples_per_epoch: 20000,
            ..NetOptions::default()
        },
    );

    network.set_pause_duration(3.0);

    println!("===============================");
    network.show_me();
    println!("===============================");
    println!("Training...");

    network.sdg(&data);
}
