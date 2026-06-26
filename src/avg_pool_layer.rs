use ndarray::Array2;

use crate::base_layer::*;
use crate::max_pool_layer::PoolLayerConfig;

pub struct AveragePoolLayer {
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

impl AveragePoolLayer {
    pub fn new(config: &PoolLayerConfig) -> Self {
        let (channels, input_h, input_w) = config.input;
        let (pool_h, pool_w) = config.pool_size;
        let stride = config.stride;

        let (out_h, out_w) = config.get_output_size();
        let (input_size, output_size) = config.get_n_in_n_out();

        Self {
            base: BaseLayer {
                input_size,
                output_size,
                // These fields are not used for Pool layers
                weights: Array2::zeros((0, 0)),
                biases: Array2::zeros((0, 0)),
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

impl Layer for AveragePoolLayer {
    fn get_base(&self) -> &BaseLayer {
        &self.base
    }

    fn get_base_mut(&mut self) -> &mut BaseLayer {
        &mut self.base
    }

    fn get_name(&self) -> String {
        format!(
            "AveragePoolLayer({}×{}, stride={})",
            self.pool_h, self.pool_w, self.stride
        )
    }

    fn clone_layer(&self) -> Box<dyn Layer> {
        Box::new(Self {
            base: self.base.clone(),
            channels: self.channels,
            input_h: self.input_h,
            input_w: self.input_w,
            pool_h: self.pool_h,
            pool_w: self.pool_w,
            stride: self.stride,
            out_h: self.out_h,
            out_w: self.out_w,
        })
    }

    fn get_type(&self) -> LayerTypes {
        LayerTypes::AveragePool
    }

    // Activation is identity (average-pool has no activation function)
    fn activate(&self, z: &Array2<f64>) -> Array2<f64> {
        z.clone()
    }

    fn activate_prime(&self, z: &Array2<f64>) -> Array2<f64> {
        Array2::ones(z.dim())
    }

    fn forward(&self, input: &Array2<f64>, _is_training: bool) -> ForwardData {
        let mut output = Array2::<f64>::zeros((self.channels * self.out_h * self.out_w, 1));

        let scale = 1.0 / ((self.pool_h * self.pool_w) as f64);

        for c in 0..self.channels {
            for oh in 0..self.out_h {
                for ow in 0..self.out_w {
                    let mut sum_val = 0.0;
                    for ph in 0..self.pool_h {
                        for pw in 0..self.pool_w {
                            let ih = oh * self.stride + ph;
                            let iw = ow * self.stride + pw;
                            let idx = c * self.input_h * self.input_w + ih * self.input_w + iw;
                            sum_val += input[[idx, 0]];
                        }
                    }
                    let out_idx = c * self.out_h * self.out_w + oh * self.out_w + ow;
                    output[[out_idx, 0]] = sum_val * scale;
                }
            }
        }

        ForwardData {
            activation: output,
            ..ForwardData::default()
        }
    }

    fn backward(
        &self,
        _input: &Array2<f64>,
        output_error: &Array2<f64>,
        _forward_data: &ForwardData,
    ) -> BackwardData {
        let mut input_grad = Array2::<f64>::zeros((self.channels * self.input_h * self.input_w, 1));

        let scale = 1.0 / (self.pool_h * self.pool_w) as f64;

        for c in 0..self.channels {
            for oh in 0..self.out_h {
                for ow in 0..self.out_w {
                    let out_idx = c * self.out_h * self.out_w + oh * self.out_w + ow;
                    let grad = output_error[[out_idx, 0]] * scale;
                    for ph in 0..self.pool_h {
                        for pw in 0..self.pool_w {
                            let ih = oh * self.stride + ph;
                            let iw = ow * self.stride + pw;
                            let idx = c * self.input_h * self.input_w + ih * self.input_w + iw;
                            input_grad[[idx, 0]] += grad;
                        }
                    }
                }
            }
        }

        BackwardData {
            input_gradient: input_grad,
            nabla_b: Array2::zeros((0, 0)),
            nabla_w: Array2::zeros((0, 0)),
        }
    }
}

#[rustfmt::skip]
#[cfg(test)]
mod tests {
    use ndarray::array;
    use super::*;

    #[test]
    fn test_forward() {
        let layer = AveragePoolLayer::new(&PoolLayerConfig {
            input: (1, 4, 4),
            pool_size: (2, 2),
            stride: 2,
        });
        let input = array![
            [2.0], [8.0], [2.0], [0.0],
            [4.0], [6.0], [5.0], [1.0],
            [7.0], [3.0], [8.0], [3.0],
            [1.0], [9.0], [8.0], [3.0]
        ]; // shape = (16, 1)
        let output = layer.forward(&input, false).activation;
        let expected = array![[5.0], [2.0], [5.0], [5.5]]; // shape = (4, 1)
        assert_eq!(output, expected);
    }

    #[test]
    fn test_backward() {
        let layer = AveragePoolLayer::new(&PoolLayerConfig {
            input: (1, 4, 4),
            pool_size: (2, 2),
            stride: 2,
        });
        let output_error = array![[4.0], [8.0], [12.0], [16.0]]; // shape = (4, 1)
        let bw = layer.backward(&Array2::zeros((0, 0)), &output_error, &ForwardData::default());
        let expected = array![
            [1.0], [1.0], [2.0], [2.0],
            [1.0], [1.0], [2.0], [2.0],
            [3.0], [3.0], [4.0], [4.0],
            [3.0], [3.0], [4.0], [4.0]
        ]; // shape = (16, 1)
        assert_eq!(bw.input_gradient, expected);
    }
}
