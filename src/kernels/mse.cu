#include "cuda_utils.cuh"
#include "mse.cuh"
#include <math.h>

__global__ void mse_grad(const float *d_output, const float *d_target,
                         float *d_grad, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < n) {
    d_grad[idx] = 2.0f * (d_output[idx] - d_target[idx]) / n;
  }
}

__global__ void mse_loss(const float *d_output, const float *d_target,
                         float *d_loss, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int tid = threadIdx.x;

  __shared__ float partial[THREADS_PER_BLOCK];
  __syncthreads();

  if (idx < n) {
    float diff = d_output[idx] - d_target[idx];
    partial[tid] = diff * diff;
  } else {
    partial[tid] = 0.0f;
  }
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (tid < stride)
      partial[tid] += partial[tid + stride];
    __syncthreads(); // partial[0] holds sum of block
  }

  if (tid == 0) {
    atomicAdd(d_loss, partial[0]);
  }
}