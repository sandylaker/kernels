extern crate openblas_src;
use ndarray::{Array2, ArrayBase, Data, Ix2};
use num_traits::Float;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;

pub fn polynomial_kernel<S, A>(
    x: &ArrayBase<S, Ix2>,
    y: &ArrayBase<S, Ix2>,
    gamma: A,
    constant: A,
    degree: A,
) -> Array2<A>
where
    S: Data<Elem = A> + Send + Sync,
    A: Float + Send + Sync + Clone + 'static,
{
    let mut dot_product: Array2<A> = x.dot(&y.t());
    dot_product.par_mapv_inplace(|z| (gamma * z + constant).powf(degree));
    dot_product
}

#[pyfunction(name = "polynomial_kernel")]
pub fn polynomial_kernel_py<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    y: PyReadonlyArray2<f64>,
    gamma: f64,
    constant: f64,
    degree: f64,
) -> &'py PyArray2<f64> {
    let x = x.as_array();
    let y = y.as_array();
    let result = polynomial_kernel(&x, &y, gamma, constant, degree);
    result.into_pyarray(py)
}

#[cfg(test)]
mod tests {
    use super::polynomial_kernel;
    use ::approx::assert_abs_diff_eq;
    use ndarray::{array, Array, Array2};

    #[test]
    fn test_polynomial_kernel() {
        let x: Array2<f64> = Array::range(0.0, 20.0, 1.0).into_shape((5, 4)).unwrap();
        let y: Array2<f64> = Array::range(10.0, 30.0, 1.0).into_shape((5, 4)).unwrap();
        let gamma = 0.1f64;
        let consant = 2.0f64;
        let degree = -1.0f64;

        let result = polynomial_kernel(&x, &y, gamma, consant, degree);

        let expected: Array2<f64> = array![
            [0.10638298, 0.08474576, 0.07042254, 0.06024096, 0.05263158],
            [0.03597122, 0.0273224, 0.02202643, 0.01845018, 0.01587302],
            [0.02164502, 0.01628664, 0.01305483, 0.01089325, 0.00934579],
            [0.01547988, 0.01160093, 0.00927644, 0.00772798, 0.00662252],
            [0.01204819, 0.00900901, 0.00719424, 0.00598802, 0.00512821]
        ];
        assert_abs_diff_eq!(result, &expected, epsilon = 1e-4);
    }
}
