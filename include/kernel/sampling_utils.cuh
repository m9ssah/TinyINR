#pragma once

#include <cstdint>
#include <limits>
#include <stdexcept>

#include "kernel/cuda_utils.cuh"

// Shared checks for the fixed-size, default-stream sampling kernels.
namespace gpu_sampling {
inline void require(bool condition, const char *message) {
  if (!condition) throw std::invalid_argument(message);
}

inline int count(int64_t a, int64_t b) {
  const int64_t limit = std::numeric_limits<int>::max() - THREADS_PER_BLOCK;
  require(a > 0 && b > 0 && a <= limit / b,
          "Sampling dimensions exceed signed-int kernel limits");
  return static_cast<int>(a * b);
}

inline void devicePointer(const void *pointer) {
  require(pointer != nullptr, "Sampling pointer must not be null");
  cudaPointerAttributes attributes{};
  if (cudaPointerGetAttributes(&attributes, pointer) != cudaSuccess) {
    cudaGetLastError();
    throw std::invalid_argument("Sampling pointer is not a device allocation");
  }
  int device;
  CUDA_CHECK(cudaGetDevice(&device));
  require(attributes.type == cudaMemoryTypeDevice && attributes.device == device,
          "Sampling buffers must be on the current CUDA device");
}

template <typename T> T *allocate(size_t elements) {
  require(elements > 0 && elements <= std::numeric_limits<size_t>::max() / sizeof(T),
          "Sampling allocation size overflow");
  T *result;
  CUDA_CHECK(cudaMalloc(&result, elements * sizeof(T)));
  return result;
}

inline void resetStatus(int *status) {
  CUDA_CHECK(cudaMemset(status, 0, sizeof(int)));
}

inline void checkStatus(const int *status) {
  int value;
  CUDA_CHECK(cudaMemcpy(&value, status, sizeof(value), cudaMemcpyDeviceToHost));
  if (value) throw std::runtime_error("Invalid GPU sampling data or numerical overflow");
}
} // namespace gpu_sampling
