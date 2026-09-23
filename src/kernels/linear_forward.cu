#include "kernel/linear.cuh"

__global__ void linear_forward_kernel(const float *d_input,
                                      const float *d_weight_T,
                                      const float *d_bias, float *d_output,
                                      int rows, int in_dim, int out_dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < rows * out_dim) {
    int r = idx / out_dim; // row index
    int o = idx % out_dim; // output dimension index
    float sum = d_bias[o];
    for (int i = 0; i < in_dim; i++) {
      sum += d_input[r * in_dim + i] * d_weight_T[i * out_dim + o];
    }
    d_output[r * out_dim + o] = sum;
  }
}

// must run after device weight has been updated to keep the transposed copy in
// sync
__global__ void transpose_weight_kernel(const float *d_weight,
                                        float *d_weight_T, int in_dim,
                                        int out_dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < in_dim * out_dim) {
    int o = idx / in_dim; // weight is [out_dim, in_dim]
    int i = idx % in_dim;
    d_weight_T[i * out_dim + o] = d_weight[idx];
  }
}