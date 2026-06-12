extern crate blas_src;

pub mod cnn_net;
pub mod conv_layer;
pub mod max_pool_layer;
pub mod sigmoid_layer;
pub mod softmax_layer;
pub mod types;

pub use base_net::{NetOptions, FeedForwardNet};
pub use base_layer::Layer;

mod base_net;
mod base_layer;
mod box_muller;
mod relu;
mod sigmoid;
mod softmax;
mod utils;
mod im2col;
mod col2im;

// Suppress panic output for all tests
#[cfg(test)]
#[ctor::ctor(unsafe)]
fn init_test() {
    std::panic::set_hook(Box::new(|_| {}));
}
