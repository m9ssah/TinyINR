#pragma once

__global__ void silu_kernel(const float *d_input, float *d_output, int n);

__global__ void silu_backward_kernel(const float *d_pre_activation,
                                     const float *d_upstream, float *d_output,
                                     int n);

__device__ float sigmoid(float x);

__host__ void cpu_silu(const float *input, float *output, int n);