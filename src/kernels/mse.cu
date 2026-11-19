#include "kernel/cuda_utils.cuh"
#include "kernel/mse.cuh"
#include <math.h>

__global__ void mse_grad_kernel(const float *d_output, const float *d_target,
                                float *d_grad, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < n) {
    d_grad[idx] = 2.0f * (d_output[idx] - d_target[idx]) / n;
  }
}

__global__ void mse_loss_kernel(const float *d_output, const float *d_target,
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
void gpuMseLossAndGrad(const float *d_output, const float *d_target,
                       float *d_loss, float *d_grad, int n) {
  CUDA_CHECK(cudaMemset(d_loss, 0, sizeof(float)));
  mse_loss_kernel<<<compute_grid_size(n), THREADS_PER_BLOCK>>>(
      d_output, d_target, d_loss, n);
  CUDA_CHECK_LAST_ERROR();
  mse_grad_kernel<<<compute_grid_size(n), THREADS_PER_BLOCK>>>(
      d_output, d_target, d_grad, n);
  CUDA_CHECK_LAST_ERROR();
}
