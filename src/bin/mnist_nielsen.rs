use simple_conv_network::{load_mnist, ActivationFn};
use simple_conv_network::network::*;
use simple_conv_network::conv_pool_layer::*;
use simple_conv_network::fully_connected_layer::*;
use simple_conv_network::softmax_layer::SoftmaxLayer;

//
// http://neuralnetworksanddeeplearning.com/chap6.html
//
// >>> net = Network([
//         ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28),
//                       filter_shape=(20, 1, 5, 5),
//                       poolsize=(2, 2),
//                       activation_fn=ReLU),
//         ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12),
//                       filter_shape=(40, 20, 5, 5),
//                       poolsize=(2, 2),
//                       activation_fn=ReLU),
//         FullyConnectedLayer(n_in=40*4*4, n_out=100, activation_fn=ReLU),
//         SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)
// >>> net.SGD(training_data, 60, mini_batch_size, 0.03,
//             validation_data, test_data, lmbda=0.1)
//
// With the above architecture, Michael Nielsen obtained a classification accuracy of 99.23%
// Let's see if we can achieve similar results with our implementation.
//

fn main() {
    let mnist_data = load_mnist(50000).expect("Failed to load MNIST dataset");
    println!(
        "Training data size: {} samples, {} validation samples, {} test samples",
        mnist_data.training.len(),
        mnist_data.test.len(),
        mnist_data.validation.len()
    );

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
            max_epochs: 60,
            mini_batch_size: 10,
            eta: 0.03,
            regularization_l2: 0.1,
            stop_early: false,
            ..NetOptions::default()
        },
    );

    println!("===============================");
    network.show_me();
    println!("===============================");
    println!("Training...");

    network.sdg(&mnist_data);
}