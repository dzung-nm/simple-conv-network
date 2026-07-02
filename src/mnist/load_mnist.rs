use byteorder::{ByteOrder, LittleEndian};
use ndarray::{Array1, Array2};
use std::fs::File;
use std::fs::exists;
use std::io::Read;

use super::unzip::unzip;
use super::augment::new_augmented_data;
use crate::types::*;

// Assure that you can see this folder.
const DATA_DIR: &str = "data";

struct LabelData {
    count: u32,
    data: Vec<u8>,
}

fn load_labels(file_path: &str) -> std::io::Result<LabelData> {
    let mut file = File::open(file_path)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;

    // The first 4 bytes contain the count (u32)
    // In our case [80, 195, 0, 0] means 0xC350 = 50000 labels
    let count = LittleEndian::read_u32(&buffer[0..4]);

    Ok(LabelData {
        count,
        data: buffer[4..].to_vec(),
    })
}

struct ImageData {
    count: u32,
    dims: u32,
    data: Vec<f64>,
}

fn load_images(file_path: &str) -> std::io::Result<ImageData> {
    let mut file = File::open(file_path)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;

    // The first 8 bytes contain the count and dimensions (both u32)
    let count = LittleEndian::read_u32(&buffer[0..4]);
    let dims = LittleEndian::read_u32(&buffer[4..8]);

    let total_elements = (count as usize) * (dims as usize);
    let mut data = Vec::with_capacity(total_elements);

    let pixel_bytes = &buffer[8..];
    for chunk in pixel_bytes.chunks_exact(4).take(total_elements) {
        let value_f32 = LittleEndian::read_f32(chunk);
        data.push(value_f32 as f64);
    }

    Ok(ImageData { count, dims, data })
}

struct MnistDataset {
    images: Array2<f64>, // shape [N, 784]
    labels: Array1<u8>,  // shape [N]
}

fn build_dataset(name: &str) -> MnistDataset {
    let labels_path = format!("{}/{}-labels.bin", DATA_DIR, name);
    let images_path = format!("{}/{}-images.bin", DATA_DIR, name);

    let labels = load_labels(&labels_path).expect("Failed to load labels");
    let images = load_images(&images_path).expect("Failed to load images");

    assert_eq!(
        labels.count, images.count,
        "Labels and images counts do not match"
    );

    let images_shape = (images.count as usize, images.dims as usize);
    let labels_shape = labels.count as usize;

    MnistDataset {
        labels: Array1::from_shape_vec(labels_shape, labels.data).unwrap(),
        images: Array2::from_shape_vec(images_shape, images.data).unwrap(),
    }
}

fn one_hot_label(label: u8) -> Array2<f64> {
    let mut one_hot = Array2::zeros((10, 1));
    one_hot[[label as usize, 0]] = 1.0;
    one_hot
}

pub fn load_mnist() -> std::io::Result<Dataset> {
    let data_files = [
        format!("{}/train-images.bin", DATA_DIR),
        format!("{}/train-labels.bin", DATA_DIR),
        format!("{}/test-images.bin", DATA_DIR),
        format!("{}/test-labels.bin", DATA_DIR),
        format!("{}/validation-images.bin", DATA_DIR),
        format!("{}/validation-labels.bin", DATA_DIR),
    ];

    if data_files.iter().all(|f| exists(f).unwrap()) {
        println!("All MNIST data files already exist. Skipping unzip.");
    } else {
        let zip_file = format!("{}/mnist.zip", DATA_DIR);
        unzip(&zip_file).expect("Failed to unzip mnist.zip");
    }

    let training = build_dataset("train");
    let validation = build_dataset("validation");
    let test = build_dataset("test");

    let training_formatted = training
        .images
        .outer_iter()
        .zip(training.labels.iter())
        .map(|(img_row, &label)| {
            let img_col = Array2::from_shape_vec((784, 1), img_row.to_vec()).unwrap();
            TrainingItem(img_col, one_hot_label(label))
        })
        .collect();

    let validation_formatted = validation
        .images
        .outer_iter()
        .zip(validation.labels.iter())
        .map(|(img_row, &label)| {
            let img_col = Array2::from_shape_vec((784, 1), img_row.to_vec()).unwrap();
            TestItem(img_col, label)
        })
        .collect();

    let test_formatted = test
        .images
        .outer_iter()
        .zip(test.labels.iter())
        .map(|(img_row, &label)| {
            let img_col = Array2::from_shape_vec((784, 1), img_row.to_vec()).unwrap();
            TestItem(img_col, label)
        })
        .collect();

    Ok(Dataset {
        training: training_formatted,
        validation: validation_formatted,
        test: test_formatted,
        dataset_type: DatasetType::Mnist,
        labels: (0..10).map(|i| i.to_string()).collect(),
        new_augmented_data: Some(new_augmented_data),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn test_load_mnist() {
        let data = load_mnist().unwrap();
        assert_eq!(data.training.len(), 50000);
        assert_eq!(data.validation.len(), 10000);
        assert_eq!(data.test.len(), 10000);
        let first_training = data.training.first().unwrap();
        let first_validation = data.validation.first().unwrap();
        assert_eq!(first_training.0.shape(), &[784, 1]);
        assert_eq!(first_training.1.shape(), &[10, 1]);
        assert_eq!(first_validation.0.shape(), &[784, 1]);
        assert_eq!(first_validation.1, 3);
        println!("All good!");
    }

    #[test]
    fn test_load_labels() {
        let train_labels_file = "data/train-labels.bin";
        if !Path::new(train_labels_file).exists() {
            println!("Skipping test_load_labels: {} not found", train_labels_file);
            return;
        }
        let labels = load_labels(train_labels_file).unwrap();
        assert_eq!(labels.count, 50000);
        assert_eq!(labels.data.len(), 50000);
    }

    #[test]
    fn test_load_images() {
        let train_image_file = "data/train-images.bin";
        if !Path::new(train_image_file).exists() {
            println!("Skipping test_load_images: {} not found", train_image_file);
            return;
        }
        let images = load_images(train_image_file).unwrap();
        assert_eq!(images.count, 50000);
        assert_eq!(images.dims, 784);
        assert_eq!(images.data.len(), 50000 * 784);
    }
}
