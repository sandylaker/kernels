mod chi2;

pub use chi2::chi2_kernel;

use numpy::ndarray::{Array2};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;


#[pyfunction(name="chi2_kernel")]
fn chi2_kernel_py<'py> (
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    y: PyReadonlyArray2<f64>,
    gamma: f64
) -> &'py PyArray2<f64> {
    let x = x.as_array();
    let y = y.as_array();
    let result = chi2_kernel(&x, &y, gamma);
    let result = Array2::from(result);
    result.into_pyarray(py)
}


#[pymodule]
#[pyo3(name="rust_kernels")]
fn rust_ext(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(chi2_kernel_py))?;

    Ok(())
}
