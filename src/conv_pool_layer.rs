/// This layer combines Convolution, ReLU, and Max Pooling into a single layer,
/// it may improve training speed a little (actually, not much as I expected)

use ndarray::{Array2, Array3, s};

use crate::base_layer::*;
use crate::box_muller::box_muller_random;
use crate::col2im::col2im;
use crate::im2col::im2col;
use crate::relu::{relu, relu_prime};

pub struct ConvPoolLayerConfig {
    pub input: (usize, usize, usize), // in_channels, input_h, input_w
    pub kernel_size: (usize, usize), // kernel_h, kernel_w
    pub num_filters: usize,
    pub stride: usize,
    pub padding: usize,
    pub pool_size: (usize, usize), // pool_h, pool_w
    pub pool_stride: usize,
}

pub struct ConvPoolLayer {
    base: BaseLayer,
    in_channels: usize,
    input_h: usize,
    input_w: usize,
    num_filters: usize,
    kernel_h: usize,
    kernel_w: usize,
    stride: usize,
    padding: usize,
    pool_h: usize,
    pool_w: usize,
    pool_stride: usize,
    conv_out_h: usize,  // Output height after convolution
    conv_out_w: usize,  // Output width after convolution
    out_h: usize,       // Output height after pooling
    out_w: usize,       // Output width after pooling
}

impl ConvPoolLayer {
    pub fn new(config: &ConvPoolLayerConfig) -> Self {
        let (in_channels, input_h, input_w) = config.input;
        let (kernel_h, kernel_w) = config.kernel_size;
        let (num_filters, stride, padding) = (config.num_filters, config.stride, config.padding);
        let (pool_h, pool_w) = config.pool_size;
        let pool_stride = config.pool_stride;

        let kernel_size = in_channels * kernel_h * kernel_w;
        let std = (2.0 / kernel_size as f64).sqrt(); // He initialization
        let weights =
            Array2::from_shape_fn((num_filters, kernel_size), |_| box_muller_random() * std);
        let biases = Array2::zeros((num_filters, 1));

        // Conv output dimensions
        let conv_out_h = (input_h + 2 * padding - kernel_h) / stride + 1;
        let conv_out_w = (input_w + 2 * padding - kernel_w) / stride + 1;

        // Pool output dimensions
        let pool_out_h = (conv_out_h - pool_h) / pool_stride + 1;
        let pool_out_w = (conv_out_w - pool_w) / pool_stride + 1;

        ConvPoolLayer {
            base: BaseLayer {
                input_size: in_channels * input_h * input_w,
                output_size: num_filters * pool_out_h * pool_out_w,
                weights,
                biases,
            },
            in_channels,
            num_filters,
            kernel_h,
            kernel_w,
            input_h,
            input_w,
            conv_out_h,
            conv_out_w,
            out_h: pool_out_h,
            out_w: pool_out_w,
            stride,
            padding,
            pool_h,
            pool_w,
            pool_stride,
        }
    }
}

impl Layer for ConvPoolLayer {
    fn get_base(&self) -> &BaseLayer {
        &self.base
    }

    fn get_base_mut(&mut self) -> &mut BaseLayer {
        &mut self.base
    }

    fn get_name(&self) -> String {
        format!(
            "ConvPoolLayer({}f, kernel={}×{}, stride={}, pad={}, pool={}×{}, pool_stride={})",
            self.num_filters, self.kernel_h, self.kernel_w, self.stride, self.padding,
            self.pool_h, self.pool_w, self.pool_stride
        )
    }

    fn clone_layer(&self) -> Box<dyn Layer> {
        Box::new(Self {
            base: self.base.clone(),
            in_channels: self.in_channels,
            num_filters: self.num_filters,
            kernel_h: self.kernel_h,
            kernel_w: self.kernel_w,
            input_h: self.input_h,
            input_w: self.input_w,
            conv_out_h: self.conv_out_h,
            conv_out_w: self.conv_out_w,
            out_h: self.out_h,
            out_w: self.out_w,
            stride: self.stride,
            padding: self.padding,
            pool_h: self.pool_h,
            pool_w: self.pool_w,
            pool_stride: self.pool_stride,
        })
    }

    fn get_type(&self) -> LayerTypes {
        LayerTypes::ConvPool
    }

    fn activate(&self, z: &Array2<f64>) -> Array2<f64> {
        relu(z)
    }

    fn activate_prime(&self, z: &Array2<f64>) -> Array2<f64> {
        relu_prime(z)
    }

    fn forward(&self, input: &Array2<f64>, _is_training: bool) -> ForwardData {
        // Step 1: Convolution using im2col
        let conv_spatial = self.conv_out_h * self.conv_out_w;

        // im2col: (in_ch × kH × kW,  conv_out_h × conv_out_w)
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

        // Conv: (num_filters, kernel_size) @ (kernel_size, conv_spatial) = (num_filters, conv_spatial)
        let mut z_2d = self.base.weights.dot(&cols);
        for i in 0..self.num_filters {
            let b = self.base.biases[[i, 0]];
            z_2d.row_mut(i).mapv_inplace(|x| x + b);
        }

        // Step 2: ReLU activation
        let a_2d = relu(&z_2d);

        // Step 3: Max pooling
        // Reshape a_2d to 3D: (num_filters, conv_out_h, conv_out_w)
        let a_3d = a_2d
            .to_shape((self.num_filters, self.conv_out_h, self.conv_out_w))
            .expect("ConvPoolLayer forward: reshape to 3D failed")
            .to_owned();

        let pool_spatial = self.out_h * self.out_w;
        let mut pooled = Array2::<f64>::zeros((self.num_filters * pool_spatial, 1));

        for c in 0..self.num_filters {
            for oh in 0..self.out_h {
                for ow in 0..self.out_w {
                    let h_start = oh * self.pool_stride;
                    let w_start = ow * self.pool_stride;
                    let h_end = usize::min(h_start + self.pool_h, self.conv_out_h);
                    let w_end = usize::min(w_start + self.pool_w, self.conv_out_w);

                    let pool_region = a_3d.slice(s![c, h_start..h_end, w_start..w_end]);
                    let max_val = pool_region.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

                    let out_idx = c * pool_spatial + oh * self.out_w + ow;
                    pooled[[out_idx, 0]] = max_val;
                }
            }
        }

        // Flatten z_2d for caching (needed for backward)
        let z_flat = Array2::from_shape_vec(
            (self.num_filters * conv_spatial, 1),
            z_2d.iter().cloned().collect(),
        )
        .expect("ConvPoolLayer forward: reshape z failed");

        ForwardData {
            z: z_flat,
            activation: pooled,
            // Cache cols, z_2d, and a_3d for backward pass
            cache: Some(LayerCache::ConvPool {
                cols,
                z_2d,
                a_3d,
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
        let (cols, z_2d, a_3d) = match &forward_data.cache {
            Some(LayerCache::ConvPool { cols, z_2d, a_3d }) => (cols, z_2d, a_3d),
            _ => panic!("ConvPoolLayer backward: expected ConvPool cache"),
        };

        let conv_spatial = self.conv_out_h * self.conv_out_w;
        let pool_spatial = self.out_h * self.out_w;

        // Step 1: Backward through max pooling
        // output_error shape: (num_filters × pool_spatial, 1)
        // Need to propagate to: (num_filters × conv_spatial, 1)
        let mut pool_grad = Array3::<f64>::zeros((self.num_filters, self.conv_out_h, self.conv_out_w));

        for c in 0..self.num_filters {
            for oh in 0..self.out_h {
                for ow in 0..self.out_w {
                    let h_start = oh * self.pool_stride;
                    let w_start = ow * self.pool_stride;
                    let h_end = usize::min(h_start + self.pool_h, self.conv_out_h);
                    let w_end = usize::min(w_start + self.pool_w, self.conv_out_w);

                    // Find argmax in this pool window
                    let mut max_val = f64::NEG_INFINITY;
                    let mut max_h = h_start;
                    let mut max_w = w_start;

                    for h in h_start..h_end {
                        for w in w_start..w_end {
                            let val = a_3d[[c, h, w]];
                            if val > max_val {
                                max_val = val;
                                max_h = h;
                                max_w = w;
                            }
                        }
                    }

                    // Propagate gradient to max position
                    let out_idx = c * pool_spatial + oh * self.out_w + ow;
                    pool_grad[[c, max_h, max_w]] += output_error[[out_idx, 0]];
                }
            }
        }

        // Reshape pool_grad to 2D: (num_filters, conv_spatial)
        let pool_grad_2d = Array2::from_shape_vec(
            (self.num_filters, conv_spatial),
            pool_grad.iter().cloned().collect(),
        )
        .expect("ConvPoolLayer backward: reshape pool_grad failed");

        // Step 2: Backward through ReLU and Conv
        // δ = pool_grad_2d ⊙ relu'(z_2d)
        let delta = pool_grad_2d * relu_prime(&z_2d);

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

#[rustfmt::skip]
#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    #[should_panic = "ConvPoolLayer backward: expected ConvPool cache"]
    fn test_backward_cache_not_exist() {
        let layer = ConvPoolLayer::new(&ConvPoolLayerConfig {
            input: (1, 4, 4),
            kernel_size: (2, 2),
            num_filters: 1,
            stride: 1,
            padding: 0,
            pool_size: (1, 1),
            pool_stride: 1,
        });
        layer.backward(&Array2::zeros((16, 1)), &Array2::zeros((9, 1)), &ForwardData::default());
    }

    #[test]
    fn test_conv_pool_layer_forward_basic() {
        // Create a simple 1-channel, 4×4 input
        let layer = ConvPoolLayer::new(&ConvPoolLayerConfig {
            input: (1, 4, 4),
            kernel_size: (2, 2),
            num_filters: 2,
            stride: 1,
            padding: 0,
            pool_size: (2, 2),
            pool_stride: 2,
        });

        let input = Array2::from_shape_vec(
            (16, 1),
            vec![
                1.0, 2.0, 3.0, 4.0,
                5.0, 6.0, 7.0, 8.0,
                9.0, 10.0, 11.0, 12.0,
                13.0, 14.0, 15.0, 16.0,
            ],
        )
        .unwrap();

        let forward_data = layer.forward(&input, false);

        // Conv output size: (4-2)/1+1 = 3×3 = 9 per filter
        // Pool output size: (3-2)/2+1 = 1×1 = 1 per filter
        // Total output: 2 filters × 1 = 2
        assert_eq!(forward_data.activation.shape(), &[2, 1]);

        // z should be conv output: 2 filters × 9 = 18
        assert_eq!(forward_data.z.shape(), &[18, 1]);
    }

    #[test]
    fn test_conv_pool_layer_backward_shapes() {
        let layer = ConvPoolLayer::new(&ConvPoolLayerConfig {
            input: (1, 4, 4),
            kernel_size: (2, 2),
            num_filters: 2,
            stride: 1,
            padding: 0,
            pool_size: (2, 2),
            pool_stride: 2,
        });

        let input = Array2::from_elem((16, 1), 1.0);
        let forward_data = layer.forward(&input, false);

        // Output error has same shape as activation
        let output_error = Array2::from_elem((2, 1), 0.5);

        let backward_data = layer.backward(&input, &output_error, &forward_data);

        // Input gradient should match input shape
        assert_eq!(backward_data.input_gradient.shape(), &[16, 1]);

        // Nabla_w should match weights shape: (2 filters, 1×2×2 = 4)
        assert_eq!(backward_data.nabla_w.shape(), &[2, 4]);

        // Nabla_b should match biases shape: (2, 1)
        assert_eq!(backward_data.nabla_b.shape(), &[2, 1]);
    }

    #[test]
    fn test_conv_pool_layer_output_dimensions() {
        // Test various configurations
        let configs = vec![
            // (input, kernel, filters, stride, pad, pool, pool_stride, expected_output)
            ((1, 8, 8), (3, 3), 4, 1, 0, (2, 2), 2, 4 * 3 * 3), // (8-3+0)/1+1=6, (6-2)/2+1=3
            ((3, 28, 28), (5, 5), 6, 1, 2, (2, 2), 2, 6 * 14 * 14), // (28+4-5)/1+1=28, (28-2)/2+1=14
        ];

        for (input, kernel, filters, stride, pad, pool, pool_stride, expected_out) in configs {
            let layer = ConvPoolLayer::new(&ConvPoolLayerConfig {
                input,
                kernel_size: kernel,
                num_filters: filters,
                stride,
                padding: pad,
                pool_size: pool,
                pool_stride,
            });

            assert_eq!(
                layer.base.output_size,
                expected_out,
                "Failed for config: input={:?}, kernel={:?}, filters={}, stride={}, pad={}, pool={:?}, pool_stride={}",
                input, kernel, filters, stride, pad, pool, pool_stride
            );
        }
    }
}
