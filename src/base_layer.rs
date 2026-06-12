use ndarray::Array2;

#[derive(Debug, PartialEq)]
pub enum LayerTypes {
    Sigmoid,
    Softmax,
    Conv,
    MaxPool,
}

pub struct LayerData {
    pub z: Array2<f64>,
    pub activation: Array2<f64>,
}

pub struct BaseLayer {
    pub input_size: usize,
    pub output_size: usize,
    pub weights: Array2<f64>,
    pub biases: Array2<f64>,
    pub nabla_w: Array2<f64>,
    pub nabla_b: Array2<f64>,
}

impl BaseLayer {
    pub fn reset_gradients(&mut self) {
        self.nabla_w.fill(0.0);
        self.nabla_b.fill(0.0);
    }
}

pub trait Layer {
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

    fn forward(&self, input: &Array2<f64>) -> LayerData {
        let base = self.get_base();
        let z = base.weights.dot(input) + &base.biases;
        let activation = self.activate(&z);
        LayerData { z, activation }
    }

    /// input: a_l-1, output_error: δ_l
    fn backward(&mut self, input: &Array2<f64>, output_error: &Array2<f64>) -> LayerData {
        // Maybe we can avoid this recomputation by storing z somewhere in the forward pass?
        let z = {
            let base = self.get_base();
            base.weights.dot(input) + &base.biases
        };

        let delta = output_error * self.activate_prime(&z);

        let base = self.get_base_mut();
        base.nabla_b += &delta;
        base.nabla_w += &delta.dot(&input.t());

        // Propagated error for the previous layer: W_l^T · δ_l
        let propagated = base.weights.t().dot(&delta);

        LayerData {
            z,
            activation: propagated,
        }
    }
}
