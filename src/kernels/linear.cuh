#pragma once

__global__ void linear_forward_kernel(const float *d_input,
                                      const float *d_weight,
                                      const float *d_bias, float *d_output,
                                      int rows, int in_dim, int out_dim);
