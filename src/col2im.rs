use ndarray::Array2;

/// Inverse of im2col: accumulate column gradients back to input space.
/// Returns shape `(C × H × W, 1)`.
pub fn col2im(
    cols: &Array2<f64>,
    in_channels: usize,
    input_h: usize,
    input_w: usize,
    kernel_h: usize,
    kernel_w: usize,
    stride: usize,
    padding: usize,
) -> Array2<f64> {
    let out_h = (input_h + 2 * padding - kernel_h) / stride + 1;
    let out_w = (input_w + 2 * padding - kernel_w) / stride + 1;
    let mut grad = Array2::<f64>::zeros((in_channels * input_h * input_w, 1));

    for c in 0..in_channels {
        for kh in 0..kernel_h {
            for kw in 0..kernel_w {
                let row = c * kernel_h * kernel_w + kh * kernel_w + kw;
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let col = oh * out_w + ow;
                        let ih = (oh * stride + kh) as i64 - padding as i64;
                        let iw = (ow * stride + kw) as i64 - padding as i64;
                        if ih >= 0 && ih < input_h as i64 && iw >= 0 && iw < input_w as i64 {
                            let idx = c * input_h * input_w + ih as usize * input_w + iw as usize;
                            grad[[idx, 0]] += cols[[row, col]];
                        }
                    }
                }
            }
        }
    }
    grad
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_col2im() {
        let cols = array![
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
        ];
        let output = col2im(&cols, 1, 3, 3, 2, 2, 1, 0);
        let expected = array![
            [1.0], [2.0], [1.0],
            [2.0], [4.0], [2.0],
            [1.0], [2.0], [1.0],
        ]; // shape = (9, 1)
        assert_eq!(output, expected);
    }
}
