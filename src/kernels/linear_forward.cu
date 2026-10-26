#include "kernel/linear.cuh"

__global__ void linear_forward_kernel(const float *d_input,
                                      const float *d_weight_T,
                                      const float *d_bias, float *d_output,
                                      int rows, int in_dim, int out_dim) {
  __shared__ float As[TILE_SIZE][TILE_SIZE];
  __shared__ float Ws[TILE_SIZE][TILE_SIZE];

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int r = blockIdx.y * TILE_SIZE + ty; // output row index
  const int o = blockIdx.x * TILE_SIZE + tx; // output index

  float sum = (o < out_dim) ? d_bias[o] : 0.0f; // initialize with bias

  for (int k0 = 0; k0 < in_dim; k0 += TILE_SIZE) {
    const int a_col = k0 + tx;
    const int w_row = k0 + ty;

    // every thread loads one eelement of each tile
    As[ty][tx] =
        (r < rows && a_col < in_dim) ? d_input[r * in_dim + a_col] : 0.0f;
    Ws[ty][tx] = (w_row < in_dim && o < out_dim)
                     ? d_weight_T[w_row * out_dim + o]
                     : 0.0f;
    __syncthreads();

    for (int j = 0; j < TILE_SIZE; j++)
      sum += As[ty][j] * Ws[j][tx];
    __syncthreads();
  }

  if (r < rows && o < out_dim)
    d_output[r * out_dim + o] = sum;
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
