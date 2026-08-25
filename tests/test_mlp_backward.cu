#include <cassert>
#include <iostream>
#include <random>
#include <vector>

#include "model/mlp.h"
#include "training/loss.h"

#include "cuda_utils.cuh"
#include "mlp_backward.cuh"
#include "mlp_forward.cuh"
#include "mse.cuh"
#include "sgd.cuh"

static void fill_random(Tensor &t, std::mt19937 &rng) {
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (int64_t i = 0; i < t.numel(); ++i) {
    t.data()[static_cast<size_t>(i)] = dist(rng);
  }
}

static void compare(const Tensor &cpu, const Tensor &gpu, const char *what,
                    int layer, float rel) {
  std::cout << "  layer " << layer << " " << what << ": ";
  assert(check_parity_rel(cpu.data(), gpu.data(), static_cast<int>(cpu.numel()),
                          rel, 1e-5f));
}

static void run_case(const MlpConfig &config, uint32_t seed, int rows,
                     float rel, const char *name) {
  const float lr = 0.01f;
  const size_t in_count = static_cast<size_t>(rows) * config.input_dim;
  const size_t out_count = static_cast<size_t>(rows) * config.output_dim;

  Mlp cpu_model = createMlp(config, seed);
  Mlp gpu_model = createMlp(config, seed);

  std::mt19937 rng(seed);
  Tensor input({rows, config.input_dim});
  Tensor target({rows, config.output_dim});
  fill_random(input, rng);
  fill_random(target, rng);

  zeroGrad(cpu_model);
  MlpForwardResult fwd = mlpForward(cpu_model, input);
  MseLossResult loss = mseLossAndGradient(fwd.output, target);
  mlpBackward(cpu_model, fwd.cache, loss.output_gradient);
  sgdStep(cpu_model, lr);

  GpuMlp gpu = uploadMlp(gpu_model);
  GpuForwardCache cache = allocForwardCache(config, rows);
  float *d_target = cuda_alloc(out_count);

  gpuZeroGrad(gpu);
  cuda_h2d(cache.input, input.data(), in_count);
  cuda_h2d(d_target, target.data(), out_count);

  gpuMlpForward(gpu, cache);
  mse_grad_kernel<<<compute_grid_size(static_cast<int>(out_count)),
             THREADS_PER_BLOCK>>>(cache.output, d_target, cache.grad_output,
                                  static_cast<int>(out_count));
  CUDA_CHECK_LAST_ERROR();
  gpuMlpBackward(gpu, cache);
  gpuSgdStep(gpu, lr);
  CUDA_CHECK(cudaDeviceSynchronize());

  downloadMlp(gpu, gpu_model);

  std::cout << name << "\n";
  for (int i = 0; i < 4; i++) {
    compare(cpu_model.layers[i].grad_weight, gpu_model.layers[i].grad_weight,
            "grad_weight", i, rel);
    compare(cpu_model.layers[i].grad_bias, gpu_model.layers[i].grad_bias,
            "grad_bias", i, rel);
    compare(cpu_model.layers[i].weight, gpu_model.layers[i].weight, "weight", i,
            rel);
    compare(cpu_model.layers[i].bias, gpu_model.layers[i].bias, "bias", i, rel);
  }

  CUDA_CHECK(cudaFree(d_target));
  freeForwardCache(cache);
  freeGpuMlp(gpu);
}

int main() {
  MlpConfig tiny{2, 3, 2, 3, ActivationKind::SiLU};
  run_case(tiny, 42, 5, 1e-4f, "tiny 2-3-2");

  run_case(makeBaselineMlpConfig(34, 3), 42, 256, 1e-3f, "baseline 34-3");

  std::cout << "test_mlp_backward_cuda PASS\n";
  return 0;
}
