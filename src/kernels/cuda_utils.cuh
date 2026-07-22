#pragma once

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// usage example: CUDA_CHECK(cudaMalloc(...));
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t _err = (call);                                                 \
    if (_err != cudaSuccess) {                                                 \
      fprintf(stderr, "[CUDA ERROR] %s:%d  %s\n", __FILE__, __LINE__,          \
              cudaGetErrorString(_err));                                       \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

// usage example: CUDA_CHECK_LAST_ERROR()
#define CUDA_CHECK_LAST_ERROR()                                                \
  do {                                                                         \
    cudaError_t _err = cudaGetLastError();                                     \
    if (_err != cudaSuccess) {                                                 \
      fprintf(stderr, "[KERNEL LAUNCH ERROR] %s:%d  %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(_err));                                       \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

// default num, can be overwritten
#ifndef THREADS_PER_BLOCK
#define THREADS_PER_BLOCK 256
#endif

inline int compute_grid_size(int total_elements,
                             int threads_per_block = THREADS_PER_BLOCK) {
  return (total_elements + threads_per_block - 1) / threads_per_block;
}

inline float *cuda_alloc(size_t count) {
  float *ptr = nullptr;
  CUDA_CHECK(cudaMalloc((void **)&ptr, count * sizeof(float)));
  return ptr;
}

inline void cuda_h2d(float *d_dst, const float *h_src, size_t count) {
  CUDA_CHECK(
      cudaMemcpy(d_dst, h_src, count * sizeof(float), cudaMemcpyHostToDevice));
}

inline void cuda_d2h(float *h_dst, const float *d_src, size_t count) {
  CUDA_CHECK(
      cudaMemcpy(h_dst, d_src, count * sizeof(float), cudaMemcpyDeviceToHost));
}

inline bool check_parity(const float *cpu_out, const float *gpu_out, int count,
                         float tol = 1e-5f) {
  double max_err = 0.0;
  for (int i = 0; i < count; i++) {
    double diff = fabs((double)cpu_out[i] - (double)gpu_out[i]);
    if (diff > max_err)
      max_err = diff;
    if (diff > tol) {
      fprintf(stderr, "[PARITY FAIL] index %d: cpu=%f  gpu=%f  diff=%e\n", i,
              cpu_out[i], gpu_out[i], diff);
      return false;
    }
  }
  printf("[PARITY PASS] max absolute error = %e\n", max_err);
  return true;
}
