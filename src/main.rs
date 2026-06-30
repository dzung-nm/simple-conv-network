use simple_conv_network::{load_mnist, ActivationFn};
use simple_conv_network::conv_pool_layer::*;
use simple_conv_network::network::*;
use simple_conv_network::fully_connected_layer::FullyConnectedLayer;
use simple_conv_network::softmax_layer::SoftmaxLayer;

fn main() {
    let data = load_mnist(50000).expect("Failed to load MNIST dataset");

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
            Box::new(FullyConnectedLayer::with_dropout(40 * 4 * 4, 100, ActivationFn::ReLU, 0.5)),
            Box::new(FullyConnectedLayer::with_activation(100, 30, ActivationFn::Sigmoid)),
            Box::new(SoftmaxLayer::new(30, 10)),
        ],
        NetOptions {
            max_epochs: 20,
            mini_batch_size: 20,
            eta: 0.1,
            regularization_l2: 0.1,
            stop_early: false,
            ..NetOptions::default()
        },
    );

    // You will see more details about the training process, including training/test accuracy
    // for each epoch. But it will slow down the training process.
    // Comment it out will speed up the training.
    network.log_more();

    println!("===============================");
    network.show_me();
    println!("===============================");
    println!("Training...");

    network.sdg(&data);
}
