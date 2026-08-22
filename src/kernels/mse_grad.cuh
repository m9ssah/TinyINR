#pragma once

__global__ void mse_grad(const float *d_output, const float *d_target,
                         float *d_grad, int n);

                         