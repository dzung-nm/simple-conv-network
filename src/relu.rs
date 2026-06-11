use ndarray::Array2;

pub fn relu(z: &Array2<f64>) -> Array2<f64> {
    z.mapv(|x| x.max(0.0))
}

pub fn relu_prime(z: &Array2<f64>) -> Array2<f64> {
    z.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relu() {
        let z = Array2::from_shape_vec((2, 2), vec![-2.0, -0.5, 0.0, 3.0]).unwrap();
        let a = relu(&z);
        assert_eq!(a[[0, 0]], 0.0);
        assert_eq!(a[[0, 1]], 0.0);
        assert_eq!(a[[1, 0]], 0.0);
        assert_eq!(a[[1, 1]], 3.0);
    }

    #[test]
    fn test_relu_prime() {
        let z = Array2::from_shape_vec((2, 2), vec![-2.0, -0.5, 0.0, 3.0]).unwrap();
        let d = relu_prime(&z);
        assert_eq!(d[[0, 0]], 0.0);
        assert_eq!(d[[0, 1]], 0.0);
        assert_eq!(d[[1, 0]], 0.0); // 0 is not > 0
        assert_eq!(d[[1, 1]], 1.0);
    }
}

