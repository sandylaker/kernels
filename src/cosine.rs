use super::pairwise_kernel::pairwise_kernel;
use ndarray::{Array2, ArrayBase, ArrayView1, Data, Ix2, Zip};
use num_traits::Float;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;

pub fn cosine_kernel<S, A>(x: &ArrayBase<S, Ix2>, y: &ArrayBase<S, Ix2>) -> Array2<A>
where
    S: Data<Elem = A> + Send + Sync,
    A: Float + Sync + Send,
{
    let cosine_pairwise_fn = |x_single: ArrayView1<A>, y_single: ArrayView1<A>| {
        let (inner, x_sq, y_sq) = Zip::from(x_single).and(y_single).par_fold(
            || (A::zero(), A::zero(), A::zero()),
            |acc, x_feat, y_feat| {
                let (mut acc_inner, mut acc_x_sq, mut acc_y_sq) = acc;
                acc_inner = acc_inner + *x_feat * (*y_feat);
                acc_x_sq = acc_x_sq + *x_feat * (*x_feat);
                acc_y_sq = acc_y_sq + *y_feat * (*y_feat);
                return (acc_inner, acc_x_sq, acc_y_sq);
            },
            |acc, other_acc| {
                let (mut acc_inner, mut acc_x_sq, mut acc_y_sq) = acc;
                let (other_acc_inner, other_acc_x_sq, other_acc_y_sq) = other_acc;
                acc_inner = acc_inner + other_acc_inner;
                acc_x_sq = acc_x_sq + other_acc_x_sq;
                acc_y_sq = acc_y_sq + other_acc_y_sq;
                return (acc_inner, acc_x_sq, acc_y_sq);
            },
        );

        let x_norm = x_sq.sqrt() + A::from(1e-8 as f64).unwrap();
        let y_norm = y_sq.sqrt() + A::from(1e-8 as f64).unwrap();

        inner / (x_norm * y_norm)
    };

    pairwise_kernel(x, y, cosine_pairwise_fn)
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
