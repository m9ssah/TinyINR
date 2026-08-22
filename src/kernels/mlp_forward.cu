#include "mlp_forward.cuh"

#include "cuda_utils.cuh"
#include "linear.cuh"
#include "silu.cuh"

void gpuMlpForward(const GpuMlp &mlp, const GpuForwardCache &cache) {
  const int rows = cache.rows;
  const float *current = cache.input;

  for (int i = 0; i < 3; i++) {
    const GpuLinearLayer &layer = mlp.layers[i];
    const int n = rows * layer.out_dim;

    linear_forward_kernel<<<compute_grid_size(n), THREADS_PER_BLOCK>>>(
        current, layer.weight, layer.bias, cache.pre_activation[i], rows,
        layer.in_dim, layer.out_dim);
    CUDA_CHECK_LAST_ERROR();

    silu_kernel<<<compute_grid_size(n), THREADS_PER_BLOCK>>>(
        cache.pre_activation[i], cache.activation[i], n);
    CUDA_CHECK_LAST_ERROR();

    current = cache.activation[i];
  }

  const GpuLinearLayer &last = mlp.layers[3];
  linear_forward_kernel<<<compute_grid_size(rows * last.out_dim),
                          THREADS_PER_BLOCK>>>(current, last.weight, last.bias,
                                               cache.output, rows, last.in_dim,
                                               last.out_dim);
  CUDA_CHECK_LAST_ERROR();
}
