use ndarray::{Array2, ArrayBase, ArrayView1, Axis, Data, Ix2, Zip};
use num_traits::Float;

pub fn pairwise_kernel<S, A, F>(
    x: &ArrayBase<S, Ix2>,
    y: &ArrayBase<S, Ix2>,
    pairwise_fn: F,
) -> Array2<A>
where
    S: Data<Elem = A> + Send + Sync,
    A: Float + Sync + Send,
    F: Fn(ArrayView1<A>, ArrayView1<A>) -> A + Sync + Send,
{
    let num_x_samples = x.len_of(Axis(0));
    let num_y_samples = y.len_of(Axis(0));
    let mut result: Array2<A> = Array2::zeros((num_x_samples, num_y_samples));

    Zip::from(result.rows_mut())
        .and(x.rows())
        .par_for_each(|result_row, x_single| {
            // for a fix single x sample, iterate through y samples.
            Zip::from(result_row)
                .and(y.rows())
                .par_for_each(|result_single, y_single| {
                    // for a fix single x sample and a single y sample, compute the pairwise, and assign it to the
                    // corresponding entry in the result matrix.
                    let kernel_element = pairwise_fn(x_single, y_single);
                    *result_single = kernel_element;
                })
        });

    result
}
