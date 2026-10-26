#include "kernel/cicfm.cuh"

__global__ void cicfm_assembly_kernel(const float *d_features,
                                      const float *d_z0, const float *d_z1,
                                      const float *d_t, float *d_zt,
                                      float *d_target_velocity, float *d_input,
                                      int rows, int feature_dim, int channels) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= rows) {
    return;
  }

  const int in_dim = feature_dim + channels + 1;
  const float time = d_t[idx];

  // input row layout: [features | zt | t]
  for (int f = 0; f < feature_dim; f++) {
    d_input[idx * in_dim + f] = d_features[idx * feature_dim + f];
  }

  for (int c = 0; c < channels; c++) {
    const float z0 = d_z0[idx * channels + c];
    const float z1 = d_z1[idx * channels + c];
    const float zt = (1.0f - time) * z0 + time * z1;

    d_zt[idx * channels + c] = zt;
    d_target_velocity[idx * channels + c] = z1 - z0;
    d_input[idx * in_dim + feature_dim + c] = zt;
  }

  d_input[idx * in_dim + feature_dim + channels] = time;
}
