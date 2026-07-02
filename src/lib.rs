#[cfg(target_os = "macos")]
extern crate blas_src;

pub mod network;
pub mod conv_layer;
pub mod conv_pool_layer;
pub mod max_pool_layer;
pub mod avg_pool_layer;
pub mod fully_connected_layer;
pub mod softmax_layer;
pub mod types;
pub mod lenet5;

pub use base_layer::{Layer, ActivationFn};
pub use mnist::load_mnist;
pub use cifar10::load_cifar10;

mod base_layer;
mod box_muller;
mod relu;
mod sigmoid;
mod softmax;
mod utils;
mod im2col;
mod col2im;
mod mnist;
mod cifar10;
mod images;
mod transforms;

// Suppress panic output for all tests
#[cfg(test)]
#[ctor::ctor(unsafe)]
fn init_test() {
    std::panic::set_hook(Box::new(|_| {}));
}
