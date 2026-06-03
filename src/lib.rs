extern crate blas_src;

mod box_muller;
mod sigmoid;
mod unzip;
mod utils;

pub mod network;
pub mod load_mnist;

// Suppress panic output for all tests
#[cfg(test)]
#[ctor::ctor(unsafe)]
fn init_test() {
    std::panic::set_hook(Box::new(|_| {}));
}
