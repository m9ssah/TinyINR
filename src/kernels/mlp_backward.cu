#include "mlp_backward.cuh"

#include "cuda_utils.cuh"
#include "linear.cuh"
#include "silu.cuh"

void gpuMlpBackward(const GpuMlp &mlp, const GpuForwardCache &cache) {
  const int rows = cache.rows;
  for (int i = 3; i >= 0; i--) {
    const GpuLinearLayer &layer = mlp.layers[i];

    float *upstream = (i == 3) ? cache.grad_output : cache.grad_z;
    float *input = (i == 0) ? cache.input : cache.activation[i - 1];

    linear_backward_weight_kernel<<<compute_grid_size(rows * layer.out_dim),
                                    THREADS_PER_BLOCK>>>(
        upstream, input, layer.grad_weight, rows, layer.in_dim, layer.out_dim);
    CUDA_CHECK_LAST_ERROR();
    linear_backward_bias_kernel<<<
        compute_grid_size(layer.in_dim * layer.out_dim), THREADS_PER_BLOCK>>>(
        upstream, layer.grad_bias, rows, layer.out_dim);
    CUDA_CHECK_LAST_ERROR();

    if (i > 0) {
      linear_backward_input_kernel<<<compute_grid_size(rows * layer.in_dim),
                                     THREADS_PER_BLOCK>>>(
          upstream, layer.weight, cache.grad_a, rows, layer.in_dim,
          layer.out_dim);
      CUDA_CHECK_LAST_ERROR();

      silu_backward_kernel<<<compute_grid_size(rows * layer.in_dim),
                             THREADS_PER_BLOCK>>>(cache.pre_activation[i - 1],
                                                  cache.grad_a, cache.grad_z,
                                                  rows * layer.in_dim);
      CUDA_CHECK_LAST_ERROR();
    }
  }
}