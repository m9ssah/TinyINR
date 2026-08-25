#include <cassert>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include "model/mlp.h"
#include "training/cicfm.h"
#include "training/loss.h"
#include "training/train_step.h"

#include "cuda_utils.cuh"
#include "train_step.cuh"

namespace {

constexpr int kBatch = 2;
constexpr int kPoints = 64;
constexpr int kRows = kBatch * kPoints;
constexpr int kFeatureDim = 34; // D=2, F=8
constexpr int kChannels = 3;
constexpr int kSteps = 5;
constexpr float kLr = 0.01f;

void fill_random(Tensor &t, std::mt19937 &rng) {
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (int64_t i = 0; i < t.numel(); ++i) {
    t.data()[static_cast<size_t>(i)] = dist(rng);
  }
}

void compare(const Tensor &cpu, const Tensor &gpu, const char *what, int layer,
             float rel) {
  std::cout << "  layer " << layer << " " << what << ": ";
  assert(check_parity_rel(cpu.data(), gpu.data(), static_cast<int>(cpu.numel()),
                          rel, 1e-5f));
}

void run_mode(LossMode mode, const char *name) {
  const bool cicfm = (mode == LossMode::CICFM);
  const int input_dim =
      cicfm ? static_cast<int>(cicfmInputDim(2, kChannels, 8)) : kFeatureDim;

  const MlpConfig config = makeBaselineMlpConfig(input_dim, kChannels);
  Mlp cpu_model = createMlp(config, 42);
  Mlp gpu_model = createMlp(config, 42);

  std::mt19937 rng(123);
  Tensor features({kBatch, kPoints, kFeatureDim});
  Tensor targets({kBatch, kPoints, kChannels});
  fill_random(features, rng);
  fill_random(targets, rng);

  Tensor z0 = Tensor::zeros({kBatch, kPoints, kChannels});
  Tensor t({kBatch, kPoints, 1});
  for (int64_t i = 0; i < t.numel(); ++i) {
    t.data()[static_cast<size_t>(i)] = 0.5f;
  }

  TrainStepConfig cpu_config{mode,
                             2,
                             kChannels,
                             8,
                             kLr,
                             /*deterministic_cicfm=*/true,
                             /*seed=*/42};

  GpuTrainContext ctx =
      createGpuTrainContext(gpu_model, config, kRows, kFeatureDim, kChannels);

  std::cout << name << "\n";
  for (int step = 0; step < kSteps; ++step) {
    TrainStepResult cpu = trainStep(cpu_model, features, targets, cpu_config);
    float gpu_loss =
        gpuTrainStep(ctx, mode, features, targets, cicfm ? &z0 : nullptr,
                     cicfm ? &t : nullptr, kLr);

    // input contract parity
    if (step == 0 && cicfm) {
      CicfmBatch ref = makeDeterministicCicfmBatch(features, targets);
      std::vector<float> gpu_input(static_cast<size_t>(kRows) * input_dim);
      cuda_d2h(gpu_input.data(), ctx.cache.input, gpu_input.size());
      std::cout << "  input contract: ";
      assert(check_parity_rel(ref.input.data(), gpu_input.data(),
                              static_cast<int>(gpu_input.size()), 1e-5f,
                              1e-6f));
    }

    // per step loss parity
    std::cout << "  step " << step << " loss: cpu=" << cpu.loss
              << " gpu=" << gpu_loss << "\n";
    assert(std::fabs(cpu.loss - gpu_loss) <=
           1e-3f * std::fabs(cpu.loss) + 1e-6f);
  }

  // final model parity
  downloadMlp(ctx.mlp, gpu_model);
  for (int i = 0; i < 4; ++i) {
    compare(cpu_model.layers[i].weight, gpu_model.layers[i].weight, "weight", i,
            1e-3f);
    compare(cpu_model.layers[i].bias, gpu_model.layers[i].bias, "bias", i,
            1e-3f);
  }

  freeGpuTrainContext(ctx);
}

} // namespace

int main() {
  run_mode(LossMode::MSE, "MSE mode");
  run_mode(LossMode::CICFM, "CICFM mode");
  std::cout << "test_train_step_cuda PASS\n";
  return 0;
}
