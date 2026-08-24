#pragma once

__global__ void cicfm_assembly(const float *d_features, const float *d_z0,
                               const float *d_z1, const float *d_t, float *d_zt,
                               float *d_target_velocity, float *d_input,
                               int rows, int feature_dim, int channels);
