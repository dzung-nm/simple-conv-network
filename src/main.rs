use mnist_conv_rust::load_mnist;
use mnist_conv_rust::Layer;
use mnist_conv_rust::conv_layer::{ConvLayer, ConvLayerConfig};
use mnist_conv_rust::max_pool_layer::{MaxPoolLayer, PoolLayerConfig};
use mnist_conv_rust::network::*;
use mnist_conv_rust::sigmoid_layer::SigmoidLayer;
use mnist_conv_rust::softmax_layer::SoftmaxLayer;
use mnist_conv_rust::types::Dataset;

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
        max_epochs: 10,
        mini_batch_size: 20,
        eta: 0.1,
        regularization_l2: 5.0,
        ..NetOptions::default()
    };
    
    let conv_layer_config = ConvLayerConfig {
        input: (1, 28, 28),
        kernel_size: (5, 5),
        num_filters: 6,
        stride: 1,
        padding: 0,
    };
    
    let max_pool_layer_config = PoolLayerConfig {
        input: (6, 24, 24),
        pool_size: (2, 2),
        stride: 2,
    };
    
    let (_, max_pool_config_n_output) = max_pool_layer_config.get_n_in_n_out();

    let layers: Vec<Box<dyn Layer>> = vec![
        Box::new(ConvLayer::new(&conv_layer_config)), // → 6×24×24 = 3456
        Box::new(MaxPoolLayer::new(&max_pool_layer_config)), // → 6×12×12 = 864
        Box::new(SigmoidLayer::new(max_pool_config_n_output, 30)),
        Box::new(SoftmaxLayer::new(30, 10)),
    ];

    let mut network = Network::new(layers, net_options);

    println!("===============================");
    network.show_me();
    println!("===============================");
    println!("Training...");

    network.sdg(&data);
}
