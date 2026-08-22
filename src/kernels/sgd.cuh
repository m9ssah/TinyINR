#pragma once
#include "gpu_mlp.cuh"

__global__ void sgd(float *d_param, const float *d_grad, float lr, int n);

void gpuSgdStep(GpuMlp &gpu_mlp, float lr);
