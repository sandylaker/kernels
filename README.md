# Rust implementations of different kernel methods

## Speed Benchmark

Remark: 
* The Rust version is optimized at `O3` level. The speed is measured using the `%%timeit` magic 
command in Jupyter Notebook.
* I also tried to extract the two outer for-loops into a separate function. But it proved to slow
down the speed.

| Kernel name | Rust (Python binding) | scikit-learn (Cython) | Naive (NumPy) |
|-------------|-----------------------| --------------------- |---------------|
| Chi2        | 30 ms +/- 345 us      | 115 ms +/- 4.18 ms    | 301 ms +/- 1.85 ms |


