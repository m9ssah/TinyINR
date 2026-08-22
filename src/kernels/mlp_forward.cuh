#pragma once
#include "gpu_mlp.cuh"

// Host-side orchestrator: launches the linear/silu kernels for all four
// layers. Input is read from cache.input, result lands in cache.output.
void gpuMlpForward(const GpuMlp &mlp, const GpuForwardCache &cache);
