#pragma once

__global__ void mse_grad_kernel(const float *d_output, const float *d_target,
                         float *d_grad, int n);

__global__ void mse_loss_kernel(const float *d_output, const float *d_target,
                         float *d_loss, int n);