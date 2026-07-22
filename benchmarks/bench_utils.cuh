#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <string>
#include <vector>

#include "../src/kernels/cuda_utils.cuh"


struct GpuTimer
{
    cudaEvent_t start, stop;

    GpuTimer()
    {
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
    }

    ~GpuTimer()
    {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void start() { CUDA_CHECK(cudaEventRecord(start)); }

    void stop()
    {
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
    }

    float elapsed_ms() const
    {
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        return ms;
    }
};


