#include <algorithm>
#include <cassert>
#include <iostream>
#include <random>
#include <vector>

#include "model/mlp.h"

#include "../src/kernels/cuda_utils.cuh"
#include "../src/kernels/silu.cuh"

static void run_case(const std::vector<float> &input, const char *name) {
  const int n = static_cast<int>(input.size());

  Tensor t({n});
  std::copy(input.begin(), input.end(), t.data());
  Tensor expected = silu(t);

  std::vector<float> gpu_output(n);
  float *d_input = cuda_alloc(n);
  float *d_output = cuda_alloc(n);
  cuda_h2d(d_input, input.data(), n);

  int blocks = compute_grid_size(n);
  silu_kernel<<<blocks, THREADS_PER_BLOCK>>>(d_input, d_output, n);
  CUDA_CHECK_LAST_ERROR();
  CUDA_CHECK(cudaDeviceSynchronize());

  cuda_d2h(gpu_output.data(), d_output, n);

  std::cout << name << " n=" << n << ": ";
  assert(check_parity(expected.data(), gpu_output.data(), n, 1e-6f));

  CUDA_CHECK(cudaFree(d_input));
  CUDA_CHECK(cudaFree(d_output));
}

int main() {
  run_case({-5.0f, -1.0f, -1e-3f, 0.0f, 1e-3f, 1.0f, 5.0f}, "edge values");

  std::mt19937 rng(123);
  std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
  std::vector<float> random_input(10000);
  for (float &x : random_input) {
    x = dist(rng);
  }
  run_case(random_input, "random");

  std::cout << "test_silu PASS\n";
  return 0;
}
