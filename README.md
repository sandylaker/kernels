# A Rust implementation of different kernel methods

## Speed Benchmark

Remark: 
* The Rust version is optimized at `O3` level. The speed is measured using the `%%timeit` magic 
command in Jupyter Notebook.
* I also tried to extract the two outer for-loops into a separate function. But it proved to slow
down computation.
* Machine: 4 x Intel(R) Xeon(R) CPU @ 2.20GHz, 16GB RAM
* Input: `x = numpy.random.rand(500, 128)`, and `y = numpy.random.rand(400, 128)`. 

| Kernel name | Rust (Python binding) | scikit-learn (partially Cython) | Naive (NumPy)      |
|-------------|-----------------------|---------------------------------|--------------------|
| chi2        | 30 ms +/- 345 us      | 115 ms +/- 4.18 ms              | 301 ms +/- 1.85 ms |
| cosine | 6.18 ms +/- 27.8 us   | 2.38 ms +/- 83.9 us             | -                  |


