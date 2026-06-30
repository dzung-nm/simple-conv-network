use ndarray::{Array2, Array3};
use rand::{RngExt};

use crate::transforms::*;
use crate::types::TrainingItem;

/// Creates a new augmented data from the original data by applying random transformations.
///  * `origin' - The original dataset to augment.
///  * `multiplier` - The number of times to augment the data
///     (This number should not be greater than the number of transforms we have)
pub fn new_augmented_data(origin: &[TrainingItem], multiplier: usize) -> Vec<TrainingItem> {
    if multiplier < 1 || multiplier > 3 {
        panic!("Augmented Multiplier must be between 1 and 3, but got {}", multiplier);
    }

    let mut augmented_data = Vec::with_capacity(origin.len() * multiplier);
    let mut rng = rand::rng();

    for _ in 0..multiplier {
        for item in origin {
            let image_flat = item.0.clone().into_flat().to_vec();
            let image_3d = Array3::from_shape_vec((1, 28, 28), image_flat).unwrap();

            if rng.random_bool(0.5) {
                // Add random crop
                let cropped_image_3d = rand_crop(&image_3d, 2);
                let cropped_image = Array2::from_shape_vec(
                    (784, 1),
                    cropped_image_3d.into_flat().to_vec()
                ).unwrap();
                augmented_data.push(TrainingItem(cropped_image, item.1.clone()));
                continue;
            }

            augmented_data.push(TrainingItem(item.0.clone(), item.1.clone()));
        }
    }

    augmented_data
}
