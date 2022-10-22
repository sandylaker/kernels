extern crate openblas_src;

use ndarray::{Array2, ArrayBase, Axis, Data, Ix2, ScalarOperand};
use num_traits::Float;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;

pub fn rbf_kernel<S, A>(x: &ArrayBase<S, Ix2>, y: &ArrayBase<S, Ix2>, gamma: A) -> Array2<A>
where
    S: Data<Elem = A> + Send + Sync,
    A: Float + Send + Sync + Clone + ScalarOperand + 'static,
{
    // assume x has shape: (N, D) and y has shape (M, D),
    // then x_square has shape: (N, 1), and y_quare has shape (1, M),
    // x_y_dot has shape (N, M)
    let x_square = x.dot(&x.t()).diag().insert_axis(Axis(1)).into_owned();
    let y_square = y.dot(&y.t()).diag().insert_axis(Axis(0)).into_owned();
    let x_y_dot = x.dot(&y.t());
    // for each x_single and y_single sample pair,
    // it computes dot(x_single, x_single) - 2 * dot(x_single, y_single) + dot(y_single, y_single)
    let mut result = x_square - (x_y_dot * A::from::<f64>(2.0).unwrap()) + y_square;
    result.par_mapv_inplace(|x| (-gamma * x).exp());

    result
}

#[pyfunction(name = "rbf_kernel")]
pub fn rbf_kernel_py<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    y: PyReadonlyArray2<f64>,
    gamma: f64,
) -> &'py PyArray2<f64> {
    let x = x.as_array();
    let y = y.as_array();
    let result = rbf_kernel(&x, &y, gamma);
    result.into_pyarray(py)
}

#[cfg(test)]
mod tests {
    use super::rbf_kernel;
    use ::approx::assert_abs_diff_eq;
    use ndarray::{array, Array, Array2};

    #[test]
    fn test_polynomial_kernel() {
        let x: Array2<f64> = Array::range(0.0, 20.0, 1.0).into_shape((5, 4)).unwrap();
        let y: Array2<f64> = Array::range(10.0, 30.0, 1.0).into_shape((5, 4)).unwrap();
        let gamma = 0.002;

        let result = rbf_kernel(&x, &y, gamma);

        let expected: Array2<f64> = array![
            [0.44932896, 0.20846169, 0.07487015, 0.02081669, 0.00448059],
            [0.74976159, 0.44932896, 0.20846169, 0.07487015, 0.02081669],
            [0.96850658, 0.74976159, 0.44932896, 0.20846169, 0.07487015],
            [0.96850658, 0.96850658, 0.74976159, 0.44932896, 0.20846169],
            [0.74976159, 0.96850658, 0.96850658, 0.74976159, 0.44932896]
        ];
        assert_abs_diff_eq!(result, &expected, epsilon = 1e-4);
    }
}
