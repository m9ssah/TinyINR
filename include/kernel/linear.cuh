#pragma once

__global__ void linear_forward_kernel(const float *d_input,
                                      const float *d_weight,
                                      const float *d_bias, float *d_output,
                                      int rows, int in_dim, int out_dim);

__global__ void linear_backward_input_kernel(const float *d_upstream,
                                             const float *d_weight,
                                             float *d_grad_input, int rows,
                                             int in_dim, int out_dim);

__global__ void linear_backward_weight_kernel(const float *d_upstream,
                                              const float *d_input,
                                              float *d_grad_weight, int rows,
                                              int in_dim, int out_dim);

__global__ void linear_backward_bias_kernel(const float *d_upstream,
                                            float *d_grad_bias, int rows,
                                            int out_dim);