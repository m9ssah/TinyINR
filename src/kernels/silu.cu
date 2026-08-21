#include "silu.cuh"

#include <math.h>

static __device__ float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

__global__ void silu_kernel(const float *d_input, float *d_output, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < n) {
    float x = d_input[idx];
    d_output[idx] = x * sigmoid(x);
  }
}

__global__ void silu_backward_kernel(const float *d_pre_activation,
                                     const float *d_upstream, float *d_output,
                                     int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    const float value = d_pre_activation[idx];
    const float sig = sigmoid(value);
    const float silu_grad = sig + value * sig * (1.0f - sig);
    d_output[idx] = d_upstream[idx] * silu_grad;
  }
}
