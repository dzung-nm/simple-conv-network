/// Run the LeNet-5 architecture on the MNIST dataset.
///     cargo run --release --bin mnist_lenet5
/// It takes around 4.1s for an epoch on my machine (Mac M2 Air), and achieves
/// around 99% accuracy after 10 epochs.

use mnist_conv_rust::load_mnist;
use mnist_conv_rust::lenet5::lenet5;

fn main() {
    let mnist_data = load_mnist().expect("Failed to load MNIST dataset");
    println!(
        "Training data size: {} samples, {} validation samples, {} test samples",
        mnist_data.training.len(),
        mnist_data.test.len(),
        mnist_data.validation.len()
    );

    let mut network = lenet5();

    println!("===============================");
    network.show_me();
    println!("===============================");
    println!("Training...");

    network.sdg(&mnist_data);
}
