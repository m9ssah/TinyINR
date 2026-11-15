#pragma once
#include "nn/gpu_mlp.cuh"

void gpuMlpBackward(const GpuMlp &mlp, const GpuForwardCache &cache);