#include "mse_grad.cuh"
#include <math.h>

__global__ void mse_grad(const float *d_output, const float *d_target,
                         float *d_grad, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < n) {
    d_grad[idx] = 2.0f * (d_output[idx] - d_target[idx]) / n;
  }
}