#include "linear.cuh"

__global__ void linear_forward_kernel(const float *d_input,
                                      const float *d_weight,
                                      const float *d_bias, float *d_output,
                                      int rows, int in_dim, int out_dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < rows * out_dim) {
    int r = idx / out_dim; // row index
    int o = idx % out_dim; // output dimension index
    float sum = 0.0f;
    sum = d_bias[o];
    for (int i = 0; i < in_dim; ++i) {
      sum += d_input[r * in_dim + i] * d_weight[o * in_dim + i];
    }
    d_output[r * out_dim + o] = sum;
  }
}