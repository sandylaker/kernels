extern crate openblas_src;
use ndarray::{Array2, ArrayBase, Axis, Data, Ix2};
use num_traits::Float;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;

pub fn cosine_kernel<S, A: 'static>(x: &ArrayBase<S, Ix2>, y: &ArrayBase<S, Ix2>) -> Array2<A>
where
    S: Data<Elem = A> + Send + Sync,
    A: Float + Send + Sync + Clone,
{
    let x_sq_mat: Array2<A> = x.dot(&x.t());
    let mut x_norm = x_sq_mat.diag().insert_axis(Axis(1)).into_owned();
    x_norm.par_mapv_inplace(|z| z.sqrt());
    let x_normed = x / x_norm;

    let y_sq_mat: Array2<A> = y.dot(&y.t());
    let mut y_norm = y_sq_mat.diag().insert_axis(Axis(1)).into_owned();
    y_norm.par_mapv_inplace(|z| z.sqrt());
    let y_normed = y / y_norm;

    x_normed.dot(&y_normed.t())
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
    use ndarray::{array, Array, Array2};

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
