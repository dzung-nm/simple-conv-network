/// Return the maximum value in the input array of f64. Panics if the input array is empty.
/// Warning: if there is a f64::NAN value in the array, the result will be f64::NAN.
pub fn arr_max(a: &Vec<f64>) -> f64 {
    if a.is_empty() {
        panic!("arr_max: input array is empty");
    }
    a.iter().max_by(|x, y| x.total_cmp(y)).unwrap().clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arr_max() {
        let a = vec![4.0, 2.0, -3.0, 40.0, 0.0];
        assert_eq!(arr_max(&a), 40.0);

        let b = vec![-3.0, -2.0, 1.0, f64::MAX, 0.0, f64::MIN];
        assert_eq!(arr_max(&b), f64::MAX);

        let c = vec![-3.0, -2.0, 1.0, f64::NAN, 0.0];
        assert!(arr_max(&c).is_nan());
    }

    #[test]
    #[should_panic = "arr_max: input array is empty"]
    fn test_arr_max_is_empty() {
        let a = vec![];
        arr_max(&a);
    }
}
