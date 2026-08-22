#include <cassert>
#include <iostream>
#include <random>
#include <vector>

#include "../src/kernels/cuda_utils.cuh"
#include "../src/kernels/mlp_forward.cuh"
#include "model/mlp.h"

static void run_case(const MlpConfig &config, uint32_t seed, int rows,
                     const char *name) {
  Mlp model = createMlp(config, seed);
  Tensor input({rows, config.input_dim});
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (int64_t i = 0; i < input.numel(); ++i) {
    input.data()[static_cast<size_t>(i)] = dist(rng);
  }

  MlpForwardResult cpu = mlpForward(model, input);
  GpuMlp gpu = uploadMlp(model);
  GpuForwardCache cache = allocForwardCache(config, rows);

  const size_t in_count = static_cast<size_t>(rows) * config.input_dim;
  const size_t out_count = static_cast<size_t>(rows) * config.output_dim;

  cuda_h2d(cache.input, input.data(), in_count);
  gpuMlpForward(gpu, cache);
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> gpu_output(out_count);
  cuda_d2h(gpu_output.data(), cache.output, out_count);

  std::cout << name << ": ";
  assert(check_parity_rel(cpu.output.data(), gpu_output.data(),
                          static_cast<int>(out_count), 1e-4f, 1e-5f));

  freeForwardCache(cache);
  freeGpuMlp(gpu);
}

int main() {
  MlpConfig tiny{2, 3, 2, 3, ActivationKind::SiLU};
  run_case(tiny, 42, 5, "tiny 2-3-2");

  run_case(makeBaselineMlpConfig(34, 3), 42, 256, "baseline 34-3");

  std::cout << "test_mlp_forward_cuda PASS\n";
  return 0;
}