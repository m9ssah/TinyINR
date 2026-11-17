#pragma once

#include <cmath>
#include <stdexcept>
#include <vector>

#include "kernel/sampling_utils.cuh"

namespace sampling_test {
inline void check(bool condition, const char *message) {
  if (!condition) throw std::runtime_error(message);
}

template <typename F> void rejects(F action) {
  try { action(); }
  catch (const std::invalid_argument &) { return; }
  catch (const std::runtime_error &) { return; }
  throw std::logic_error("Expected validation failure");
}

inline void near(double actual, double expected, double absolute = 1e-6,
                 double relative = 1e-5) {
  check(std::isfinite(actual) && std::isfinite(expected) &&
        std::fabs(actual - expected) <= absolute + relative * std::fabs(expected),
        "Numerical comparison failed");
}

template <typename T> struct Buffer {
  T *data;
  size_t count;
  explicit Buffer(size_t n) : data(gpu_sampling::allocate<T>(n)), count(n) {}
  explicit Buffer(const std::vector<T> &values) : Buffer(values.size()) { upload(values); }
  ~Buffer() { cudaFree(data); }
  Buffer(const Buffer &) = delete;
  Buffer &operator=(const Buffer &) = delete;
  void upload(const std::vector<T> &values) {
    check(values.size() == count, "Test buffer size mismatch");
    CUDA_CHECK(cudaMemcpy(data, values.data(), count * sizeof(T), cudaMemcpyHostToDevice));
  }
  std::vector<T> read() const {
    std::vector<T> values(count);
    CUDA_CHECK(cudaMemcpy(values.data(), data, count * sizeof(T), cudaMemcpyDeviceToHost));
    return values;
  }
};
} // namespace sampling_test
