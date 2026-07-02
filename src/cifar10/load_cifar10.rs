use ndarray::Array2;
use std::fs::File;
use std::io::Read;

use crate::types::*;
use super::augment::new_augmented_data;

// Assure that you can see this folder.
const DATA_DIR: &str = "data/cifar10";

fn one_hot_label(label: u8) -> Array2<f64> {
    let mut one_hot = Array2::zeros((10, 1));
    one_hot[[label as usize, 0]] = 1.0;
    one_hot
}

//
// Binary version
//
// The binary version contains the files data_batch_1.bin, data_batch_2.bin, ..., data_batch_5.bin,
// as well as test_batch.bin. Each of these files is formatted as follows:
//
//   <1 x label><3072 x pixel>
//   ...
//   <1 x label><3072 x pixel>
//
// In other words, the first byte is the label of the first image, which is a number in the
// range 0-9. The next 3072 bytes are the values of the pixels of the image. The first 1024 bytes
// are the red channel values, the next 1024 the green, and the final 1024 the blue. The values are
// stored in row-major order, so the first 32 bytes are the red channel values of the first row
// of the image.
//
// Each file contains 10000 such 3073-byte "rows" of images, although there is nothing delimiting
// the rows. Therefore, each file should be exactly 30730000 bytes long.
//

/// Load CIFAR-10 data.
pub fn load_cifar10() -> std::io::Result<Dataset> {
    // Load training data from data_batch_1.bin to data_batch_5.bin
    let training_data: Vec<TrainingItem> = (1..=5)
        .map(|i| format!("{}/data_batch_{}.bin", DATA_DIR, i))
        .map(|file_path| {
            let mut file = File::open(file_path).unwrap();
            let mut buffer = Vec::new();
            file.read_to_end(&mut buffer).unwrap();
            buffer
                .chunks_exact(3073)
                .map(|chunk| {
                    let label = one_hot_label(chunk[0]);
                    let pixels = &chunk[1..]
                        .iter()
                        .map(|&x| x as f64 / 255.0)
                        .collect::<Vec<f64>>();
                    let image = Array2::from_shape_vec((3072, 1), pixels.to_vec()).unwrap();
                    TrainingItem(image, label)
                })
                .collect::<Vec<_>>()
        })
        .flatten()
        .collect::<Vec<_>>();

    // Load test data and validation data from test_batch.bin
    // We will use a half of the test data for validation and the other half for testing

    let mut file = File::open(format!("{}/test_batch.bin", DATA_DIR))?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;

    let test_data = buffer
        .chunks_exact(3073)
        .take(5000)
        .map(|chunk| {
            let pixels = &chunk[1..]
                .iter()
                .map(|&x| x as f64 / 255.0)
                .collect::<Vec<f64>>();
            let image = Array2::from_shape_vec((3072, 1), pixels.to_vec()).unwrap();
            TestItem(image, chunk[0])
        })
        .collect::<Vec<_>>();

    let validate_data = buffer
        .chunks_exact(3073)
        .skip(5000)
        .map(|chunk| {
            let pixels = &chunk[1..]
                .iter()
                .map(|&x| x as f64 / 255.0)
                .collect::<Vec<f64>>();
            let image = Array2::from_shape_vec((3072, 1), pixels.to_vec()).unwrap();
            TestItem(image, chunk[0])
        })
        .collect::<Vec<_>>();

    Ok(Dataset {
        training: training_data,
        validation: test_data,
        test: validate_data,
        dataset_type: DatasetType::Cifar10,
        labels: vec![
            "airplane",
            "automobile",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect(),
        new_augmented_data: Some(new_augmented_data),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn test_load_cifar10() {
        let start = Instant::now();
        let data = load_cifar10().unwrap();
        let duration = start.elapsed();

        println!("  Time: {:.3}ms", duration.as_secs_f64() * 1000.0);

        // Validate lengths of training, validation, and test datasets
        assert_eq!(data.training.len(), 50000);
        assert_eq!(data.validation.len(), 5000);
        assert_eq!(data.test.len(), 5000);

        // Validate shapes of the first training and validation items
        let first_training = data.training.first().unwrap();
        let first_validation = data.validation.first().unwrap();
        assert_eq!(first_training.0.shape(), &[3072, 1]);
        assert_eq!(first_training.1.shape(), &[10, 1]);
        assert_eq!(first_validation.0.shape(), &[3072, 1]);

        println!("All good!");
    }
}
