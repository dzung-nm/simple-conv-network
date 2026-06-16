use ndarray::Array2;

#[derive(Debug, PartialEq)]
pub enum LayerTypes {
    Sigmoid,
    Softmax,
    Conv,
    MaxPool,
    AveragePool,
}

pub enum LayerCache {
    Conv {
        cols: Array2<f64>,
        z_2d: Array2<f64>,
    },
}

/// Data returned from forward pass
pub struct ForwardData {
    pub z: Array2<f64>,          // Pre-activation value (needed for backward pass)
    pub activation: Array2<f64>, // Post-activation value
    pub cache: Option<LayerCache>,
}

impl ForwardData {
    pub fn dummy() -> Self {
        ForwardData {
            z: Array2::zeros((0, 0)),
            activation: Array2::zeros((0, 0)),
            cache: None,
        }
    }
}

/// Data returned from backward pass
pub struct BackwardData {
    pub input_gradient: Array2<f64>, // Gradient to propagate to the previous layer
    pub nabla_b: Array2<f64>,
    pub nabla_w: Array2<f64>,
}

pub struct BaseLayer {
    pub input_size: usize,
    pub output_size: usize,
    pub weights: Array2<f64>,
    pub biases: Array2<f64>,
}

pub trait Layer: Send + Sync {
    fn get_base(&self) -> &BaseLayer;
    fn get_base_mut(&mut self) -> &mut BaseLayer;
    fn get_name(&self) -> String;

    fn show_me(&self) {
        println!("Layer: {}", self.get_name());
        let base = self.get_base();
        println!(
            "  Input size: {}, Output size: {}",
            base.input_size, base.output_size
        );
        println!(
            "  Weights shape: {:?}, Biases shape: {:?}",
            base.weights.dim(),
            base.biases.dim()
        );
    }

    fn get_type(&self) -> LayerTypes;
    fn activate(&self, z: &Array2<f64>) -> Array2<f64>;
    fn activate_prime(&self, z: &Array2<f64>) -> Array2<f64>;

    // Default implementations for forward and backward that can be
    // applied to most layers, but can be overridden if needed (e.g., for ConvLayer, PoolLayer)

    fn forward(&self, input: &Array2<f64>) -> ForwardData {
        let base = self.get_base();
        let z = base.weights.dot(input) + &base.biases;
        let activation = self.activate(&z);
        ForwardData { z, activation, cache: None }
    }

    fn backward(
        &self,
        input: &Array2<f64>,        // activation from previous layer
        output_error: &Array2<f64>, // error signal from next layer
        forward_data: &ForwardData, // data from forward pass (contains z, activation, and cache)
    ) -> BackwardData {
        let delta = output_error * self.activate_prime(&forward_data.z);
        let nabla_w = delta.dot(&input.t());

        // Propagated error for the previous layer: W_l^T · δ_l
        let input_gradient = self.get_base().weights.t().dot(&delta);

        BackwardData {
            input_gradient,
            nabla_b: delta,
            nabla_w,
        }
    }
}
