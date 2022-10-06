mod chi2;
mod cosine;

pub use chi2::{chi2_kernel, chi2_kernel_py};
pub use cosine::{cosine_kernel, cosine_kernel_py};

use pyo3::prelude::*;

#[pymodule]
#[pyo3(name = "rust_kernels")]
fn rust_ext(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(chi2_kernel_py))?;
    m.add_wrapped(wrap_pyfunction!(cosine_kernel_py))?;

    Ok(())
}
