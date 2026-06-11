use ndarray::Array2;

// https://petewarden.com/2015/04/20/why-gemm-is-at-the-heart-of-deep-learning/

/// Convert a column-vector input `(C × H × W, 1)` into an im2col matrix.
/// Returns shape `(C × kH × kW,  out_H × out_W)`.
pub fn im2col(
    input: &Array2<f64>,
    in_channels: usize,
    input_h: usize,
    input_w: usize,
    kernel_h: usize,
    kernel_w: usize,
    stride: usize,
    padding: usize,
) -> Array2<f64> {
    let out_h = (input_h + 2 * padding - kernel_h) / stride + 1; // e.g: (28 + 2*0 - 5) / 1 + 1 = 24
    let out_w = (input_w + 2 * padding - kernel_w) / stride + 1; // e.g: (28 + 2*0 - 5) / 1 + 1 = 24
    let rows = in_channels * kernel_h * kernel_w; // e.g: 1 * 5 * 5 = 25
    let cols = out_h * out_w; // e.g: 24 * 24 = 576

    let mut result = Array2::<f64>::zeros((rows, cols));

    for c in 0..in_channels {
        for kh in 0..kernel_h {
            for kw in 0..kernel_w {
                let row = c * kernel_h * kernel_w + kh * kernel_w + kw;
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let col = oh * out_w + ow;
                        let ih = (oh * stride + kh) as i64 - padding as i64;
                        let iw = (ow * stride + kw) as i64 - padding as i64;
                        // check padding bounds
                        if ih >= 0 && ih < input_h as i64 && iw >= 0 && iw < input_w as i64 {
                            let idx = c * input_h * input_w + ih as usize * input_w + iw as usize;
                            result[[row, col]] = input[[idx, 0]];
                        }
                        // else: zero-padding (already 0)
                    }
                }
            }
        }
    }
    result
}

#[rustfmt::skip]
#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_im2col() {
        let input = array![
            [1.0], [2.0], [3.0],
            [4.0], [5.0], [6.0],
            [7.0], [8.0], [9.0]
        ]; // shape = (9, 1)
        let output = im2col(&input, 1, 3, 3, 2, 2, 1, 0);
        let expected = array![
            [1.0, 2.0, 4.0, 5.0],
            [2.0, 3.0, 5.0, 6.0],
            [4.0, 5.0, 7.0, 8.0],
            [5.0, 6.0, 8.0, 9.0]
        ];
        assert_eq!(output.shape(), &[4, 4]);
        assert_eq!(output, expected);

        let input = array![
            [1.0],  [2.0],  [3.0],  [4.0],
            [5.0],  [6.0],  [7.0],  [8.0],
            [9.0],  [10.0], [11.0], [12.0],
            [13.0], [14.0], [15.0], [16.0]
        ];
        let output = im2col(&input, 1, 4, 4, 2, 2, 1, 0);
        let expected = array![
            [1.0, 2.0, 3.0, 5.0,  6.0,  7.0,  9.0,  10.0, 11.0],
            [2.0, 3.0, 4.0, 6.0,  7.0,  8.0,  10.0, 11.0, 12.0],
            [5.0, 6.0, 7.0, 9.0,  10.0, 11.0, 13.0, 14.0, 15.0],
            [6.0, 7.0, 8.0, 10.0, 11.0, 12.0, 14.0, 15.0, 16.0],
        ];
        assert_eq!(output.shape(), &[2 * 2, 3 * 3]);
        assert_eq!(output, expected);
    }
}
