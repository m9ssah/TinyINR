#pragma once
#include "gpu_mlp.cuh"

void gpuMlpForward(const GpuMlp &mlp, const GpuForwardCache &cache);
