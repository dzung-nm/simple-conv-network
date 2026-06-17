/// 2-D convolutional layer with ReLU activation.
use ndarray::Array2;

use crate::base_layer::*;
use crate::box_muller::box_muller_random;
use crate::col2im::col2im;
use crate::im2col::im2col;
use crate::relu::{relu, relu_prime};

pub struct ConvLayerConfig {
    pub input: (usize, usize, usize), // in_channels, input_h, input_w
    pub kernel_size: (usize, usize), // kernel_h, kernel_w
    pub num_filters: usize,
    pub stride: usize,
    pub padding: usize,
}

pub struct ConvLayer {
    base: BaseLayer,
    in_channels: usize,
    num_filters: usize,
    kernel_h: usize,
    kernel_w: usize,
    input_h: usize,
    input_w: usize,
    stride: usize,
    padding: usize,
    out_h: usize,
    out_w: usize,
}

impl ConvLayer {
    pub fn new(config: &ConvLayerConfig) -> Self {
        let (in_channels, input_h, input_w) = config.input;
        let (kernel_h, kernel_w) = config.kernel_size; 
        let (num_filters, stride, padding) = (config.num_filters, config.stride, config.padding);

        let out_h = (input_h + 2 * padding - kernel_h) / stride + 1;
        let out_w = (input_w + 2 * padding - kernel_w) / stride + 1;
        let kernel_size = in_channels * kernel_h * kernel_w;

        let std = (2.0 / kernel_size as f64).sqrt(); // He initialization
        let weights =
            Array2::from_shape_fn((num_filters, kernel_size), |_| box_muller_random() * std);
        let biases = Array2::zeros((num_filters, 1));

        ConvLayer {
            base: BaseLayer {
                input_size: in_channels * input_h * input_w,
                output_size: num_filters * out_h * out_w,
                weights,
                biases,
            },
            in_channels,
            num_filters,
            kernel_h,
            kernel_w,
            input_h,
            input_w,
            out_h,
            out_w,
            stride,
            padding,
        }
    }
}

impl Layer for ConvLayer {
    fn get_base(&self) -> &BaseLayer {
        &self.base
    }

    fn get_base_mut(&mut self) -> &mut BaseLayer {
        &mut self.base
    }

    fn get_name(&self) -> String {
        format!(
            "ConvLayer({}f, {}×{}, stride={}, pad={})",
            self.num_filters, self.kernel_h, self.kernel_w, self.stride, self.padding
        )
    }

    fn get_type(&self) -> LayerTypes {
        LayerTypes::Conv
    }

    fn activate(&self, z: &Array2<f64>) -> Array2<f64> {
        relu(z)
    }

    fn activate_prime(&self, z: &Array2<f64>) -> Array2<f64> {
        relu_prime(z)
    }

    fn forward(&self, input: &Array2<f64>, _is_training: bool) -> ForwardData {
        let spatial = self.out_h * self.out_w;

        // im2col: (in_ch × kH × kW,  out_h × out_w), e.g: (25, 576)
        let cols = im2col(
            input,
            self.in_channels,
            self.input_h,
            self.input_w,
            self.kernel_h,
            self.kernel_w,
            self.stride,
            self.padding,
        );

        let mut z_2d = self.base.weights.dot(&cols); // e.g: (4, 25) @ (25, 576) = (4, 576)
        for i in 0..self.num_filters {
            let b = self.base.biases[[i, 0]];
            z_2d.row_mut(i).mapv_inplace(|x| x + b);
        }

        // Activation: ReLU
        let a_2d = relu(&z_2d);

        // Flatten to column vectors (num_filters * spatial, 1)
        let out_size = self.num_filters * spatial;
        let z_flat = Array2::from_shape_vec((out_size, 1), z_2d.iter().cloned().collect())
            .expect("ConvLayer forward: reshape z failed");
        let a_flat = Array2::from_shape_vec((out_size, 1), a_2d.iter().cloned().collect())
            .expect("ConvLayer forward: reshape activation failed");

        ForwardData {
            z: z_flat,
            activation: a_flat,
            // Cache cols and z_2d for backward pass
            cache: Some(LayerCache::Conv {
                cols,
                z_2d,
            }),
            ..ForwardData::default()
        }
    }

    fn backward(
        &self,
        _input: &Array2<f64>,
        output_error: &Array2<f64>,
        forward_data: &ForwardData,
    ) -> BackwardData {
        let (cols, z_2d) = match &forward_data.cache {
            Some(LayerCache::Conv { cols, z_2d }) => (cols, z_2d),
            _ => panic!("ConvLayer backward: expected Conv cache"),
        };

        let spatial = self.out_h * self.out_w;

        // Reshape output_error: (num_filters × spatial, 1) → (num_filters, spatial)
        let output_error_2d = Array2::from_shape_vec(
            (self.num_filters, spatial),
            output_error.iter().cloned().collect(),
        )
        .expect("ConvLayer backward: reshape output_error failed");

        // δ = output_error_2d ⊙ relu'(z_2d) - use cached z_2d directly
        let delta = output_error_2d * relu_prime(&z_2d);

        // ∇filters (= ∇W): (num_filters, in_ch × kH × kW)
        let nabla_w = delta.dot(&cols.t());

        // ∇biases: (num_filters, 1) = sum over spatial dimension
        let nabla_b = delta
            .sum_axis(ndarray::Axis(1))
            .insert_axis(ndarray::Axis(1));

        // Propagated error: col2im(filtersᵀ @ δ)
        let delta_cols = self.base.weights.t().dot(&delta);
        let input_grad = col2im(
            &delta_cols,
            self.in_channels,
            self.input_h,
            self.input_w,
            self.kernel_h,
            self.kernel_w,
            self.stride,
            self.padding,
        );

        BackwardData {
            input_gradient: input_grad,
            nabla_b,
            nabla_w,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    #[should_panic = "ConvLayer backward: expected Conv cache"]
    fn test_backward_cache_not_exist() {
        let layer = ConvLayer::new(&ConvLayerConfig {
            input: (1, 4, 4),
            kernel_size: (2, 2),
            num_filters: 1,
            stride: 1,
            padding: 0,
        });
        layer.backward(&Array2::zeros((16, 1)), &Array2::zeros((9, 1)), &ForwardData::default());
    }
}
