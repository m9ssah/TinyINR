#include "kernel/linear.cuh"

__global__ void linear_forward_kernel(const float *d_input,
                                      const float *d_weight,
                                      const float *d_bias, float *d_output,
                                      int rows, int in_dim, int out_dim) {
  __shared__ float As[TILE_SIZE][TILE_SIZE];
  __shared__ float Ws[TILE_SIZE][TILE_SIZE];

  int tx = threadIdx.x;
  int ty = threadIdx.y; 
  int r = blockIdx.y * TILE_SIZE + ty; // output row index
  int o = blockIdx.x * TILE_SIZE + tx; // output column index
  float sum = 0.0f;

  for (int k0 = 0; k0 < in_dim; k0 += TILE_SIZE) {
    // every thread loads one eelement of each tile
    As[ty][tx] = d_input[r * in_dim + (k0 + tx)];
    Ws[ty][tx] = d_weight[o * in_dim + (k0 + ty)];
    __syncthreads();

    // every thread does TILE FMA based on shared memory
    for (int j = 0; j < TILE_SIZE; j++)
      sum += As[ty][j] * Ws[j][tx];
    __syncthreads();
  }
  // handlde bias and write back to global memory
  if (r < rows && o < out_dim)
    d_output[r * out_dim + o] = sum + d_bias[o];
}