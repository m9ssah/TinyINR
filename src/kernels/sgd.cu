#include "cuda_utils.cuh"
#include "sgd.cuh"

__global__ void sgd(float *d_param, const float *d_grad, float lr, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < n) {
    d_param[idx] -= lr * d_grad[idx];
  }
}

void gpuSgdStep(GpuMlp &gpu_mlp, float lr) {
  const int n = sizeof(gpu_mlp.layers) / sizeof(gpu_mlp.layers[0]);
  for (int i = 0; i < n; i++) {
    GpuLinearLayer &layer = gpu_mlp.layers[i];
    const int w_count = layer.in_dim * layer.out_dim;
    const int b_count = layer.out_dim;

    sgd<<<compute_grid_size(w_count), THREADS_PER_BLOCK>>>(
        layer.weight, layer.grad_weight, lr, w_count);
    CUDA_CHECK_LAST_ERROR();

    sgd<<<compute_grid_size(b_count), THREADS_PER_BLOCK>>>(
        layer.bias, layer.grad_bias, lr, b_count);
    CUDA_CHECK_LAST_ERROR();
  }
}
