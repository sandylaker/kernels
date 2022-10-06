use ndarray::{Array2, ArrayBase, Axis, Data, Ix2, Zip};
use num_traits::Float;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;

pub fn cosine_kernel<S, A: 'static>(x: &ArrayBase<S, Ix2>, y: &ArrayBase<S, Ix2>) -> Array2<A>
where
    S: Data<Elem = A> + Send + Sync,
    A: Float + Sync + Send,
{
    let num_x_samples = x.len_of(Axis(0));
    let num_y_samples = y.len_of(Axis(0));
    let mut result: Array2<A> = Array2::zeros((num_x_samples, num_y_samples));

    Zip::from(result.rows_mut())
        .and(x.rows())
        .par_for_each(|result_row, x_single| {
            // for a fix single x sample, iterate through all the y samples.
            Zip::from(result_row)
                .and(y.rows())
                .par_for_each(|result_single, y_single| {
                    // for a fix single x sample and a single y sample, compute the chi2 value, and assign it to the
                    // corresponding entry in the result matrix.
                    let inner = x_single.dot(&y_single);
                    let x_norm = (x_single.dot(&x_single)).sqrt();
                    let y_norm = (y_single.dot(&y_single)).sqrt();

                    *result_single = inner / (x_norm * y_norm);
                })
        });

    result
}

#[pyfunction(name = "cosine_kernel")]
pub fn cosine_kernel_py<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    y: PyReadonlyArray2<f64>,
) -> &'py PyArray2<f64> {
    let x = x.as_array();
    let y = y.as_array();
    let result = cosine_kernel(&x, &y);
    result.into_pyarray(py)
}

#[cfg(test)]
mod tests {
    use super::cosine_kernel;
    use approx::assert_abs_diff_eq;
    use ndarray::{array, Array, Array1, Array2};

    #[test]
    fn test_cosine_kernel() {
        let x: Array2<f64> = Array::range(0.0, 20.0, 1.0).into_shape((5, 4)).unwrap();
        let y: Array2<f64> = Array::range(10.0, 30.0, 1.0).into_shape((5, 4)).unwrap();

        let result = cosine_kernel(&x, &y);

        let expected: Array2<f64> = array![
            [0.85584885, 0.84270097, 0.83467719, 0.82927778, 0.82539834],
            [0.99463515, 0.99175012, 0.98975383, 0.98831736, 0.98724115],
            [0.99979532, 0.99898125, 0.99820794, 0.99757828, 0.99707412],
            [0.99989794, 0.99994358, 0.99967854, 0.99938445, 0.99911832],
            [0.99945175, 0.99996633, 0.99997869, 0.99986779, 0.99973164]
        ];
        assert_abs_diff_eq!(result, &expected, epsilon = 1e-4);
    }
}
