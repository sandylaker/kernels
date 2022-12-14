mod chi2;
mod cosine;
mod polynomial;
mod rbf;

pub use chi2::{chi2_kernel, chi2_kernel_py};
pub use cosine::{cosine_kernel, cosine_kernel_py};
pub use polynomial::{polynomial_kernel, polynomial_kernel_py};
pub use rbf::{rbf_kernel, rbf_kernel_py};

use pyo3::prelude::*;

#[pymodule]
#[pyo3(name = "rust_kernels")]
fn rust_ext(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(chi2_kernel_py))?;
    m.add_wrapped(wrap_pyfunction!(cosine_kernel_py))?;
    m.add_wrapped(wrap_pyfunction!(polynomial_kernel_py))?;
    m.add_wrapped(wrap_pyfunction!(rbf_kernel_py))?;

    Ok(())
}
