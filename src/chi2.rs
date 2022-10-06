use super::pairwise_kernel::pairwise_kernel;
use ndarray::{Array2, ArrayBase, ArrayView1, Data, Ix2, Zip};
use num_traits::Float;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;

pub fn chi2_kernel<S, A>(x: &ArrayBase<S, Ix2>, y: &ArrayBase<S, Ix2>, gamma: A) -> Array2<A>
where
    S: Data<Elem = A> + Send + Sync,
    A: Float + Sync + Send,
{
    let chi2_pairwise_fn = |x_single: ArrayView1<A>, y_single: ArrayView1<A>| {
        let kernel_element = Zip::from(x_single).and(y_single).par_fold(
            || A::zero(),
            |acc, x_feat, y_feat| {
                let denom: A = *x_feat - *y_feat;
                let nom: A = *x_feat + *y_feat;
                if !nom.is_zero() {
                    return acc + (denom * denom) / nom;
                }
                acc
            },
            |acc, other_acc| acc + other_acc,
        );
        let tmp: A = -gamma * kernel_element;
        tmp.exp()
    };

    pairwise_kernel(x, y, chi2_pairwise_fn)
}

#[pyfunction(name = "chi2_kernel")]
pub fn chi2_kernel_py<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    y: PyReadonlyArray2<f64>,
    gamma: f64,
) -> &'py PyArray2<f64> {
    let x = x.as_array();
    let y = y.as_array();
    let result = chi2_kernel(&x, &y, gamma);
    result.into_pyarray(py)
}

#[cfg(test)]
mod test {
    use super::chi2_kernel;
    use approx::assert_abs_diff_eq;
    use ndarray::{array, Array, Array2};

    #[test]
    fn test_chi2_kernel() {
        let x: Array2<f64> = Array::range(0.0, 20.0, 1.0).into_shape((5, 4)).unwrap();
        let y: Array2<f64> = Array::range(10.0, 30.0, 1.0).into_shape((5, 4)).unwrap();

        let result = chi2_kernel(&x, &y, 1.0);

        let expected: Array2<f64> = array![
            [
                1.66529257e-14,
                4.11812542e-21,
                7.73000784e-28,
                1.24623573e-34,
                1.83088295e-41
            ],
            [
                1.80202638e-04,
                4.28700807e-09,
                1.86259781e-14,
                2.98540324e-20,
                2.53233176e-26
            ],
            [
                4.62685801e-01,
                3.00735806e-03,
                9.41383015e-07,
                4.31006232e-11,
                5.39587047e-16
            ],
            [
                5.24563839e-01,
                5.74051233e-01,
                1.24775590e-02,
                1.93942642e-05,
                4.68403060e-09
            ],
            [
                6.76948903e-03,
                6.14410810e-01,
                6.47898437e-01,
                2.95198283e-02,
                1.34906848e-04
            ]
        ];
        assert_abs_diff_eq!(result, &expected, epsilon = 1e-4);
    }
}
