#undef NDEBUG
#include <cassert>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

#include "model/mlp.h"

#include "kernel/cuda_utils.cuh"
#include "kernel/train_step.cuh"

namespace {

// 128 rows * 3 channels = 384 loss elements = 2 blocks of mse_loss_kernel
// when we have 3+ blocks, the atomic order might change the last bits run to
// run
constexpr int kRows = 128;
constexpr int kFeatureDim = 34;
constexpr int kChannels = 3;
constexpr int kSteps = 10;

struct Data {
  Tensor features{{1, kRows, kFeatureDim}};
  Tensor targets{{1, kRows, kChannels}};
};

Data makeData() {
  Data data;
  std::mt19937 rng(5);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (int64_t i = 0; i < data.features.numel(); ++i)
    data.features.data()[static_cast<size_t>(i)] = dist(rng);
  for (int64_t i = 0; i < data.targets.numel(); ++i)
    data.targets.data()[static_cast<size_t>(i)] = 0.5f * (dist(rng) + 1.0f);
  return data;
}

std::vector<float> downloadWeights(const GpuMlp &mlp) {
  size_t total_params = 0;
  for (int i = 0; i < 4; i++) {
    const GpuLinearLayer &layer = mlp.layers[i];
    total_params +=
        (static_cast<size_t>(layer.in_dim) * layer.out_dim) + layer.out_dim;
  }

  std::vector<float> all(total_params);
  size_t offset = 0;

  for (int i = 0; i < 4; i++) {
    const GpuLinearLayer &layer = mlp.layers[i];
    const size_t w_count = static_cast<size_t>(layer.in_dim) * layer.out_dim;
    const size_t b_count = static_cast<size_t>(layer.out_dim);

    cuda_d2h(all.data() + offset, layer.weight, w_count);
    offset += w_count;

    cuda_d2h(all.data() + offset, layer.bias, b_count);
    offset += b_count;
  }
  return all;
}

// catches features/targets uploaded into the wrong buffer or layout
void testStepZeroMatchesEagerPath(const MlpConfig &config, const Data &data) {
  GpuTrainContext eager = createGpuTrainContext(createMlp(config, 42), config,
                                                kRows, kFeatureDim, kChannels);
  const float eager_loss = gpuTrainStep(eager, LossMode::MSE, data.features,
                                        data.targets, nullptr, nullptr, 1e-3f);

  GpuTrainContext resident = createGpuTrainContext(
      createMlp(config, 42), config, kRows, kFeatureDim, kChannels);
  GpuAdamW adamw = createGpuAdamW(resident.mlp);
  gpuUploadTrainData(resident, data.features, data.targets);
  gpuTrainStepResident(resident, adamw, AdamWConfig{});
  const float resident_loss = gpuReadLoss(resident);

  assert(std::memcmp(&eager_loss, &resident_loss, sizeof(float)) == 0);

  freeGpuAdamW(adamw);
  freeGpuTrainContext(resident);
  freeGpuTrainContext(eager);
  std::cout << "  step-0 loss matches gpuTrainStep bit-for-bit: ok\n";
}

void testUploadOnceEqualsUploadEveryStep(const MlpConfig &config,
                                         const Data &data) {
  AdamWConfig cfg;
  cfg.lr = 1e-2f;

  GpuTrainContext once = createGpuTrainContext(createMlp(config, 7), config,
                                               kRows, kFeatureDim, kChannels);
  GpuAdamW adamw_once = createGpuAdamW(once.mlp);
  gpuUploadTrainData(once, data.features, data.targets);
  for (int step = 0; step < kSteps; step++)
    gpuTrainStepResident(once, adamw_once, cfg);

  GpuTrainContext every = createGpuTrainContext(createMlp(config, 7), config,
                                                kRows, kFeatureDim, kChannels);
  GpuAdamW adamw_every = createGpuAdamW(every.mlp);
  for (int step = 0; step < kSteps; step++) {
    gpuUploadTrainData(every, data.features, data.targets);
    gpuTrainStepResident(every, adamw_every, cfg);
  }

  const std::vector<float> w_once = downloadWeights(once.mlp);
  const std::vector<float> w_every = downloadWeights(every.mlp);
  assert(w_once.size() == w_every.size());
  assert(std::memcmp(w_once.data(), w_every.data(),
                     w_once.size() * sizeof(float)) == 0);

  freeGpuAdamW(adamw_once);
  freeGpuAdamW(adamw_every);
  freeGpuTrainContext(once);
  freeGpuTrainContext(every);
  std::cout << "  upload-once == upload-every-step after " << kSteps
            << " steps: ok\n";
}

// repeated steps on fixed data must actually fit it
void testLossDecreases(const MlpConfig &config, const Data &data) {
  GpuTrainContext ctx = createGpuTrainContext(createMlp(config, 3), config,
                                              kRows, kFeatureDim, kChannels);
  GpuAdamW adamw = createGpuAdamW(ctx.mlp);
  gpuUploadTrainData(ctx, data.features, data.targets);

  gpuTrainStepResident(ctx, adamw, AdamWConfig{});
  const float first = gpuReadLoss(ctx);
  for (int step = 1; step < 200; step++)
    gpuTrainStepResident(ctx, adamw, AdamWConfig{});
  const float last = gpuReadLoss(ctx);

  std::cout << "  loss " << first << " -> " << last << " over 200 steps\n";
  assert(last < 0.1f * first);

  freeGpuAdamW(adamw);
  freeGpuTrainContext(ctx);
}

void testUploadRejectsMismatches(const MlpConfig &config, const Data &data) {
  const MlpConfig cicfm_config =
      makeBaselineMlpConfig(kFeatureDim + 4, kChannels);
  GpuTrainContext cicfm_ctx = createGpuTrainContext(
      createMlp(cicfm_config, 1), cicfm_config, kRows, kFeatureDim, kChannels);
  bool threw = false;
  try {
    gpuUploadTrainData(cicfm_ctx, data.features, data.targets);
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  assert(threw);
  freeGpuTrainContext(cicfm_ctx);

  GpuTrainContext ctx = createGpuTrainContext(createMlp(config, 1), config,
                                              kRows, kFeatureDim, kChannels);
  Tensor short_targets({1, kRows - 1, kChannels});
  threw = false;
  try {
    gpuUploadTrainData(ctx, data.features, short_targets);
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  assert(threw);
  freeGpuTrainContext(ctx);
  std::cout << "  upload rejects CICFM-width model and shape mismatch: ok\n";
}

} // namespace

int main() {
  const MlpConfig config = makeBaselineMlpConfig(kFeatureDim, kChannels);
  const Data data = makeData();

  testStepZeroMatchesEagerPath(config, data);
  testUploadOnceEqualsUploadEveryStep(config, data);
  testLossDecreases(config, data);
  testUploadRejectsMismatches(config, data);

  std::cout << "test_train_step_resident_cuda PASS\n";
  return 0;
}
