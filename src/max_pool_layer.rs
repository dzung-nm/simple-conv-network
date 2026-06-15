use ndarray::Array2;

use crate::base_layer::*;

pub struct PoolLayerConfig {
    pub input: (usize, usize, usize), // in_channels, input_h, input_w
    pub pool_size: (usize, usize), // pool_h, pool_w
    pub stride: usize,
}

impl PoolLayerConfig {
    pub fn get_output_size(&self) -> (usize, usize) {
        let (_, input_h, input_w) = self.input;
        let (pool_h, pool_w) = self.pool_size;
        let stride = self.stride;

        let out_h = (input_h - pool_h) / stride + 1;
        let out_w = (input_w - pool_w) / stride + 1;

        (out_h, out_w)
    }

    // Returns (n_in, n_out) for the BaseLayer of this MaxPoolLayer.
    pub fn get_n_in_n_out(&self) -> (usize, usize) {
        let (channels, input_h, input_w) = self.input;
        let (out_h, out_w) = self.get_output_size();
        (channels * input_h * input_w, channels * out_h * out_w)
    }
}

pub struct MaxPoolLayer {
    base: BaseLayer,
    channels: usize, // number of feature maps (filters)
    input_h: usize,
    input_w: usize,
    pool_h: usize,
    pool_w: usize,
    stride: usize,
    out_h: usize,
    out_w: usize,
}

impl MaxPoolLayer {
    pub fn new(config: &PoolLayerConfig) -> Self {
        let (channels, input_h, input_w) = config.input;
        let (pool_h, pool_w) = config.pool_size;
        let stride = config.stride;

        let (out_h, out_w) = config.get_output_size();
        let (input_size, output_size) = config.get_n_in_n_out();

        MaxPoolLayer {
            base: BaseLayer {
                input_size,
                output_size,
                // These fields are not used for Pool layers
                weights: Array2::zeros((0, 0)),
                biases: Array2::zeros((0, 0)),
                nabla_w: Array2::zeros((0, 0)),
                nabla_b: Array2::zeros((0, 0)),
            },
            channels,
            input_h,
            input_w,
            pool_h,
            pool_w,
            stride,
            out_h,
            out_w,
        }
    }
}

impl Layer for MaxPoolLayer {
    fn get_base(&self) -> &BaseLayer {
        &self.base
    }

    fn get_base_mut(&mut self) -> &mut BaseLayer {
        &mut self.base
    }

    fn get_name(&self) -> String {
        format!(
            "MaxPoolLayer({}×{}, stride={})",
            self.pool_h, self.pool_w, self.stride
        )
    }

    fn get_type(&self) -> LayerTypes {
        LayerTypes::MaxPool
    }

    // Activation is identity (max-pool has no activation function)
    fn activate(&self, z: &Array2<f64>) -> Array2<f64> {
        z.clone()
    }

    fn activate_prime(&self, z: &Array2<f64>) -> Array2<f64> {
        Array2::ones(z.dim())
    }

    fn forward(&mut self, input: &Array2<f64>) -> LayerData {
        let mut output = Array2::<f64>::zeros((self.channels * self.out_h * self.out_w, 1));

        for c in 0..self.channels {
            for oh in 0..self.out_h {
                for ow in 0..self.out_w {
                    let mut max_val = f64::NEG_INFINITY;
                    for ph in 0..self.pool_h {
                        for pw in 0..self.pool_w {
                            let ih = oh * self.stride + ph;
                            let iw = ow * self.stride + pw;
                            let idx = c * self.input_h * self.input_w + ih * self.input_w + iw;
                            if input[[idx, 0]] > max_val {
                                max_val = input[[idx, 0]];
                            }
                        }
                    }
                    let out_idx = c * self.out_h * self.out_w + oh * self.out_w + ow;
                    output[[out_idx, 0]] = max_val;
                }
            }
        }

        // Store input in z so backward can locate the argmax
        LayerData {
            z: input.clone(),
            activation: output,
        }
    }

    /// Todo: we can optimize backward by caching the argmax positions during forward pass
    fn backward(&mut self, input: &Array2<f64>, output_error: &Array2<f64>) -> LayerData {
        let mut input_grad = Array2::<f64>::zeros((self.channels * self.input_h * self.input_w, 1));

        for c in 0..self.channels {
            for oh in 0..self.out_h {
                for ow in 0..self.out_w {
                    // Recompute argmax in this pool window
                    let mut max_val = f64::NEG_INFINITY;
                    let mut max_idx = 0usize;
                    for ph in 0..self.pool_h {
                        for pw in 0..self.pool_w {
                            let ih = oh * self.stride + ph;
                            let iw = ow * self.stride + pw;
                            let idx = c * self.input_h * self.input_w + ih * self.input_w + iw;
                            if input[[idx, 0]] > max_val {
                                max_val = input[[idx, 0]];
                                max_idx = idx;
                            }
                        }
                    }
                    let out_idx = c * self.out_h * self.out_w + oh * self.out_w + ow;
                    // if stride < pool size, so accumulate gradients for tied max positions
                    input_grad[[max_idx, 0]] += output_error[[out_idx, 0]];
                }
            }
        }

        let dummy_z = Array2::<f64>::zeros((self.channels * self.out_h * self.out_w, 1));
        LayerData {
            z: dummy_z,
            activation: input_grad,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_forward() {
        let mut layer = MaxPoolLayer::new(&PoolLayerConfig {
            input: (1, 4, 4),
            pool_size: (2, 2),
            stride: 2,
        });
        let input = array![
            [1.0], [2.0], [3.0], [4.0],
            [5.0], [6.0], [7.0], [8.0],
            [9.0], [10.0], [11.0], [12.0],
            [13.0], [14.0], [15.0], [16.0]
        ]; // shape = (16, 1)
        let output = layer.forward(&input).activation;
        let expected = array![
            [6.0], [8.0],
            [14.0], [16.0]
        ]; // shape = (4, 1)
        assert_eq!(output, expected);
    }

    #[test]
    fn test_backward() {
        let mut layer = MaxPoolLayer::new(&PoolLayerConfig {
            input: (1, 4, 4),
            pool_size: (2, 2),
            stride: 2,
        });
        let input = array![
            [1.0], [2.0], [3.0], [4.0],
            [5.0], [6.0], [7.0], [8.0],
            [9.0], [10.0], [11.0], [12.0],
            [13.0], [14.0], [15.0], [16.0]
        ];
        let output_error = array![
            [1.0], [2.0],
            [3.0], [4.0]
        ]; // shape = (4, 1)
        let input_grad = layer.backward(&input, &output_error).activation;
        let expected = array![
            [0.0], [0.0], [0.0], [0.0],
            [0.0], [1.0], [0.0], [2.0],
            [0.0], [0.0], [0.0], [0.0],
            [0.0], [3.0], [0.0], [4.0]
        ]; // shape = (16, 1)
        assert_eq!(input_grad, expected);
    }
}
