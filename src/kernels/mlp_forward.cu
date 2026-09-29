#include "kernel/cuda_utils.cuh"
#include "kernel/linear.cuh"
#include "kernel/mlp_forward.cuh"
#include "kernel/silu.cuh"


// linear_forward_kernel is tiled: one block per TILE_SIZE x TILE_SIZE patch of
// the output, so it needs a 2D launch. silu_kernel stays elementwise and 1D.
static dim3 tiled_grid(int rows, int out_dim) {
  return dim3((out_dim + TILE_SIZE - 1) / TILE_SIZE,
              (rows + TILE_SIZE - 1) / TILE_SIZE);
}

void gpuMlpForward(const GpuMlp &mlp, const GpuForwardCache &cache) {
  const int rows = cache.rows;
  const float *current = cache.input;
  const dim3 tiled_block(TILE_SIZE, TILE_SIZE);

  for (int i = 0; i < 3; i++) {
    const GpuLinearLayer &layer = mlp.layers[i];
    const int n = rows * layer.out_dim;

    linear_forward_kernel<<<tiled_grid(rows, layer.out_dim), tiled_block>>>(
        current, layer.weight_T, layer.bias, cache.pre_activation[i], rows,
        layer.in_dim, layer.out_dim);
    CUDA_CHECK_LAST_ERROR();

    silu_kernel<<<compute_grid_size(n), THREADS_PER_BLOCK>>>(
        cache.pre_activation[i], cache.activation[i], n);
    CUDA_CHECK_LAST_ERROR();

    current = cache.activation[i];
  }

  const GpuLinearLayer &last = mlp.layers[3];
  linear_forward_kernel<<<tiled_grid(rows, last.out_dim), tiled_block>>>(
      current, last.weight_T, last.bias, cache.output, rows, last.in_dim,
      last.out_dim);
  CUDA_CHECK_LAST_ERROR();
}
