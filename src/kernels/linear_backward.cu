#include "linear.cuh"

__global__ void linear_backward_input_kernel(const float *d_upstream,
                                             const float *d_weight,
                                             float *d_grad_input, int rows,
                                             int in_dim, int out_dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < rows * in_dim) {
    int r = idx / in_dim;
    int i = idx % in_dim;
    float sum = 0.0f;
    for (int o = 0; o < out_dim; o++) {
      sum += d_upstream[r * out_dim + o] * d_weight[o * in_dim + i];
    }
    d_grad_input[r * in_dim + i] = sum;
  }
}

__global__ void linear_backward_weight_kernel(const float *d_upstream,
                                              const float *d_input,
                                              float *d_grad_weight, int rows,
                                              int in_dim, int out_dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < out_dim * in_dim) {
    int o = idx / in_dim;
    int i = idx % in_dim;
    float sum = 0.0f;
    for (int r = 0; r < rows; r++) {
      sum += d_upstream[r * out_dim + o] * d_input[r * in_dim + i];
    }
    d_grad_weight[o * in_dim + i] += sum;
  }
}

__global__ void linear_backward_bias_kernel(const float *d_upstream,
                                            float *d_grad_bias, int rows,
                                            int out_dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < out_dim) {
    float sum = 0.0f;
    for (int r = 0; r < rows; r++) {
      sum += d_upstream[r * out_dim + idx];
    }
    d_grad_bias[idx] += sum;
  }
}