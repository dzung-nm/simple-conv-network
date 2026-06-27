/// This is a neural network implementation for the CIFAR-10 dataset using a simple
/// convolutional network architecture.
///     cargo run --release --bin cifar10

use simple_conv_network::conv_layer::{ConvLayer, ConvLayerConfig};
use simple_conv_network::{load_cifar10, Layer};
use simple_conv_network::avg_pool_layer::AveragePoolLayer;
use simple_conv_network::fully_connected_layer::FullyConnectedLayer;
use simple_conv_network::max_pool_layer::PoolLayerConfig;
use simple_conv_network::network::{NetOptions, Network};
use simple_conv_network::softmax_layer::SoftmaxLayer;

fn main() {
    let data = load_cifar10(10000).expect("Failed to load CIFAR-10 dataset");

    println!(
        "Training data size: {} samples, {} validation samples, {} test samples",
        data.training.len(),
        data.test.len(),
        data.validation.len()
    );

    let conv_layer_config1 = ConvLayerConfig {
        input: (3, 32, 32),
        kernel_size: (5, 5),
        num_filters: 6,
        stride: 1,
        padding: 0,
    };
    let pool_layer_config1 = PoolLayerConfig {
        input: (6, 28, 28),
        pool_size: (2, 2),
        stride: 2,
    };

    let conv_layer_config2 = ConvLayerConfig {
        input: (6, 14, 14),
        kernel_size: (5, 5),
        num_filters: 16,
        stride: 1,
        padding: 0,
    };
    let pool_layer_config2 = PoolLayerConfig {
        input: (16, 10, 10),
        pool_size: (2, 2),
        stride: 2,
    };

    let layers: Vec<Box<dyn Layer>> = vec![
        Box::new(ConvLayer::new(&conv_layer_config1)), // → 6×28×28
        Box::new(AveragePoolLayer::new(&pool_layer_config1)), // → 6×14×14
        Box::new(ConvLayer::new(&conv_layer_config2)), // → 16×10×10
        Box::new(AveragePoolLayer::new(&pool_layer_config2)), // → 16×5×5
        Box::new(FullyConnectedLayer::new(16 * 5 * 5, 120)),
        Box::new(FullyConnectedLayer::new(120, 84)),
        Box::new(SoftmaxLayer::new(84, 10)),
    ];

    let mut network = Network::new(
        layers,
        NetOptions {
            max_epochs: 30,
            mini_batch_size: 20,
            eta: 0.1,
            regularization_l2: 5.0,
            ..NetOptions::default()
        },
    );
    // network.log_more();

    println!("===============================");
    network.show_me();
    println!("===============================");
    println!("Training...");

    network.sdg(&data);
}
