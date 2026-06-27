/// Run the LeNet-5 architecture on the MNIST dataset.
///     cargo run --release --bin mnist_lenet5
/// It takes around 4.4s for an epoch on my machine (Mac M2 Air), and achieves
/// around 99% accuracy after 10-20 epochs.

use simple_conv_network::load_mnist;
use simple_conv_network::lenet5::lenet5;

fn main() {
    let mnist_data = load_mnist(50000).expect("Failed to load MNIST dataset");
    println!(
        "Training data size: {} samples, {} validation samples, {} test samples",
        mnist_data.training.len(),
        mnist_data.test.len(),
        mnist_data.validation.len()
    );

    let mut network = lenet5(30);

    println!("===============================");
    network.show_me();
    println!("===============================");
    println!("Training...");

    network.sdg(&mnist_data);
}
