#include "kernel/adamw.cuh"

__global__ void adam_w_kernel(float *d_w, float *d_w_T, const float *d_g,
                              float *d_m, float *d_v, float lr, float b1,
                              float b2, float eps, float bc1, float bc2,
                              int in_dim, int out_dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= in_dim * out_dim)
    return;
  const int o = idx / in_dim;
  const int i = idx % in_dim;

  const float gi = d_g[idx];
  const float mi = b1 * d_m[idx] + (1.0f - b1) * gi;      // first moment
  const float vi = b2 * d_v[idx] + (1.0f - b2) * gi * gi; // second moment
  d_m[idx] = mi;
  d_v[idx] = vi;

  const float wi = d_w[idx] - lr * (mi / bc1) / (sqrtf(vi / bc2) + eps);
  d_w[idx] = wi;
  d_w_T[i * out_dim + o] = wi; 
}