use ndarray::Array2;

/// Represents a training data item with image and one-hot encoded label
pub struct TrainingItem(pub Array2<f64>, pub Array2<f64>); // (image (784,1), one-hot label (10,1))

/// Represents a test/validation data item with image and raw label
pub struct TestItem(pub Array2<f64>, pub u8); // (image (784,1), label)

/// Container for MNIST dataset splits
pub struct Dataset {
    pub training: Vec<TrainingItem>,
    pub validation: Vec<TestItem>,
    pub test: Vec<TestItem>,
}
