/// This implementation is mostly the same as the original LeNet-5 architecture, except that we use
/// sigmoid activation instead of tanh, and some details may not be exactly the same.

use crate::Layer;
use crate::avg_pool_layer::*;
use crate::conv_layer::*;
use crate::max_pool_layer::PoolLayerConfig;
use crate::network::*;
use crate::sigmoid_layer::SigmoidLayer;
use crate::softmax_layer::SoftmaxLayer;

pub fn lenet5(max_epochs: usize) -> Network {
    let conv_layer_config1 = ConvLayerConfig {
        input: (1, 28, 28),
        kernel_size: (5, 5),
        num_filters: 6,
        stride: 1,
        padding: 2,
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
        Box::new(SigmoidLayer::new(16 * 5 * 5, 120)),
        Box::new(SigmoidLayer::new(120, 84)),
        Box::new(SoftmaxLayer::new(84, 10)),
    ];

    Network::new(
        layers,
        NetOptions {
            max_epochs: max_epochs.clamp(1, 100),
            mini_batch_size: 20,
            eta: 0.1,
            regularization_l2: 5.0,
            ..NetOptions::default()
        },
    )
}
