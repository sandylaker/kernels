from setuptools import setup
from setuptools_rust import Binding, RustExtension

setup(
    rust_extensions=[RustExtension("rust_kernels.rust_kernels", binding=Binding.PyO3, debug=False)],
)
