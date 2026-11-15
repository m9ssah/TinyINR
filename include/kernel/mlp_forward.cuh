#pragma once
#include "nn/gpu_mlp.cuh"

void gpuMlpForward(const GpuMlp &mlp, const GpuForwardCache &cache);
