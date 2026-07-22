#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <string>
#include <vector>

#include "../src/kernels/cuda_utils.cuh"

struct GpuTimer {
  cudaEvent_t start, stop;

  GpuTimer() {
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
  }

  ~GpuTimer() {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }

  void start() { CUDA_CHECK(cudaEventRecord(start)); }

  void stop() {
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
  }

  float elapsed_ms() const {
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    return ms;
  }
};

struct CpuTimer {
  std::chrono::steady_clock::time_point start, stop;

  void start() { start = std::chrono::steady_clock::now(); }
  void stop() { stop = std::chrono::steady_clock::now(); }

  float elapsed_ms() const {
    return std::chrono::duration<float, std::milli>(stop - start).count();
  }
};

struct L2Flusher {
  void *buffer = nullptr;
  size_t l2_bytes = 0;

  L2Flusher() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    l2_bytes = prop.l2CacheSize;
    if (l2_bytes > 0)
      CUDA_CHECK(cudaMalloc(&buffer, l2_bytes));
  }

  ~L2Flusher() {
    if (buffer)
      cudaFree(buffer);
  }

  // call between benchmark trials
  void flush() {
    if (buffer && l2_bytes > 0) {
      CUDA_CHECK(cudaMemset(buffer, 0, l2_bytes));
      CUDA_CHECK(cudaDeviceSynchronize());
    }
  }
};

inline void cuda_warmup(int warmup_iters = 5) {
  void *dummy = nullptr;
  CUDA_CHECK(cudaMalloc(&dummy, 1024));

  for (int i = 0; i < warmup_iters; i++)
    CUDA_CHECK(cudaMemset(dummy, 0, 1024));

  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaFree(dummy));
}
