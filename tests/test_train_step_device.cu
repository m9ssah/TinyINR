#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

#include "kernel/train_step.cuh"
#include "training/cicfm.h"

namespace {

void check(bool value, const char *message) {
  if (!value) throw std::runtime_error(message);
}

template <typename Error, typename F> void rejects(F action) {
  try { action(); }
  catch (const Error &) { return; }
  throw std::runtime_error("Expected validation failure");
}

void near(float actual, float expected) {
  check(std::isfinite(actual) && std::isfinite(expected) &&
        std::fabs(actual - expected) <= 1e-5f + 1e-3f * std::fabs(expected),
        "CPU/device value mismatch");
}

void matches(const float *device, const Tensor &expected, bool exact = false) {
  Tensor actual(expected.shape());
  cuda_d2h(actual.data(), device, actual.numel());
  for (int64_t i = 0; i < actual.numel(); ++i) {
    if (exact) check(actual.data()[i] == expected.data()[i], "Device data changed");
    else near(actual.data()[i], expected.data()[i]);
  }
}

void matchesModel(GpuTrainContext &ctx, const Mlp &expected, bool exact = false) {
  for (int i = 0; i < 4; ++i) {
    matches(ctx.mlp.layers[i].weight, expected.layers[i].weight, exact);
    matches(ctx.mlp.layers[i].bias, expected.layers[i].bias, exact);
  }
}

void parity(LossMode mode) {
  const bool flow = mode == LossMode::CICFM;
  const MlpConfig config{flow ? 9 : 6, 8, 2, 3, ActivationKind::SiLU};
  Mlp cpu = createMlp(config, 42);
  auto host = createGpuTrainContext(cpu, config, 6, 6, 2);
  auto device = createGpuTrainContext(cpu, config, 6, 6, 2);
  const Tensor features = sampleUniform({2, 3, 6}, -1, 1, 1);
  const Tensor targets = sampleUniform({2, 3, 2}, 0, 1, 2);
  const Tensor z0 = sampleUniform({2, 3, 2}, -0.5f, 0.5f, 3);
  Tensor t = sampleUniform({2, 3, 1}, 0, 1, 4);
  t.data()[0] = 0; t.data()[5] = 1;
  float *df = cuda_alloc(features.numel()), *dy = cuda_alloc(targets.numel());
  float *dz = cuda_alloc(z0.numel()), *dt = cuda_alloc(t.numel());
  cuda_h2d(df, features.data(), features.numel());
  cuda_h2d(dy, targets.data(), targets.numel());
  cuda_h2d(dz, z0.data(), z0.numel());
  cuda_h2d(dt, t.data(), t.numel());
  // Poison wrapper upload buffers: the device path must consume caller inputs.
  CUDA_CHECK(cudaMemset(device.d_features, 0xff, features.numel() * sizeof(float)));
  CUDA_CHECK(cudaMemset(device.d_targets, 0xff, targets.numel() * sizeof(float)));
  for (int step = 0; step < 3; ++step) {
    const auto batch = makeCicfmBatch(features, targets, z0, t);
    const Tensor &input = flow ? batch.input : features;
    const Tensor &target = flow ? batch.target_velocity : targets;
    auto forward = mlpForward(cpu, input);
    auto loss = mseLossAndGradient(forward.output, target);
    zeroGrad(cpu);
    mlpBackward(cpu, forward.cache, loss.output_gradient);
    sgdStep(cpu, 0.01f);
    StageTimings timing{};
    const float host_loss = gpuTrainStep(host, mode, features, targets,
        flow ? &z0 : nullptr, flow ? &t : nullptr, 0.01f, &timing);
    check(std::isfinite(timing.step_ms) && timing.step_ms >= 0,
          "Invalid wrapper timing");
    const auto result = gpuTrainStepDevice(device, mode, df, dy,
        flow ? dz : nullptr, flow ? dt : nullptr, 0.01f,
        step == 0 ? &timing : nullptr);
    float sum;
    cuda_d2h(&sum, result.loss_sum, 1);
    near(host_loss, loss.loss);
    near(sum / 12, loss.loss);
    matches(result.prediction, forward.output);
    matches(result.loss_target, target);
    matches(device.cache.input, input);
    matchesModel(device, cpu);
    matchesModel(host, cpu);
    if (!flow) check(result.loss_target == dy, "MSE target must be borrowed");
  }
  freeGpuTrainContext(device);
  freeGpuTrainContext(host);
  matches(df, features, true); matches(dy, targets, true);
  matches(dz, z0, true); matches(dt, t, true);
  CUDA_CHECK(cudaFree(df)); CUDA_CHECK(cudaFree(dy));
  CUDA_CHECK(cudaFree(dz)); CUDA_CHECK(cudaFree(dt));
}

void invalidMetadata() {
  const MlpConfig config{6, 8, 2, 3, ActivationKind::SiLU};
  Mlp model = createMlp(config, 1);
  auto create = [&](const MlpConfig &c, int rows, int features, int channels) {
    auto ctx = createGpuTrainContext(model, c, rows, features, channels);
    freeGpuTrainContext(ctx);
  };
  rejects<std::invalid_argument>([&] { create(config, 0, 6, 2); });
  rejects<std::invalid_argument>([&] { create(config, 1, -1, 2); });
  rejects<std::invalid_argument>([&] { create(config, 1, 6, 3); });
  rejects<std::invalid_argument>([&] { create(config, 2147483647, 6, 2); });
  rejects<std::invalid_argument>([&] { create(config, 65535 * 16 + 1, 6, 2); });
  auto bad = config; bad.hidden_layer_count = 2;
  rejects<std::invalid_argument>([&] { create(bad, 1, 6, 2); });
  bad = config; bad.activation = static_cast<ActivationKind>(99);
  rejects<std::invalid_argument>([&] { create(bad, 1, 6, 2); });
  bad = config; bad.hidden_width = 9;
  rejects<std::invalid_argument>([&] { create(bad, 1, 6, 2); });
  model.layers[3].weight.data()[0] = std::numeric_limits<float>::quiet_NaN();
  rejects<std::invalid_argument>([&] { create(config, 1, 6, 2); });
  model = createMlp(config, 1);
  model.layers.pop_back();
  rejects<std::invalid_argument>([&] { create(config, 1, 6, 2); });
}

void invalidSteps(LossMode mode) {
  const bool flow = mode == LossMode::CICFM;
  const MlpConfig config{flow ? 9 : 6, 8, 2, 3, ActivationKind::SiLU};
  const Mlp model = createMlp(config, 5);
  auto ctx = createGpuTrainContext(model, config, 2, 6, 2);
  Tensor features = sampleUniform({1, 2, 6}, -1, 1, 6);
  Tensor target = sampleUniform({1, 2, 2}, 0, 1, 7);
  Tensor z0 = Tensor::zeros({1, 2, 2});
  Tensor t = sampleUniform({1, 2, 1}, 0, 1, 8);
  const Tensor *z = flow ? &z0 : nullptr, *time = flow ? &t : nullptr;
  auto step = [&] { gpuTrainStep(ctx, mode, features, target, z, time, 0.01f); };
  rejects<std::invalid_argument>([&] {
    gpuTrainStep(ctx, static_cast<LossMode>(99), features, target, z, time, 0.01f);
  });
  for (float lr : {0.0f, -1.0f, std::numeric_limits<float>::infinity()})
    rejects<std::invalid_argument>([&] {
      gpuTrainStep(ctx, mode, features, target, z, time, lr);
    });
  rejects<std::invalid_argument>([&] {
    gpuTrainStep(ctx, mode, features.reshape({2, 1, 6}), target, z, time, 0.01f);
  });
  rejects<std::invalid_argument>([&] {
    gpuTrainStep(ctx, mode, features, target, flow ? nullptr : &z0, time, 0.01f);
  });
  rejects<std::invalid_argument>([&] {
    gpuTrainStepDevice(ctx, mode, nullptr, ctx.d_targets,
        flow ? ctx.d_z0 : nullptr, flow ? ctx.d_t : nullptr, 0.01f);
  });
  rejects<std::invalid_argument>([&] {
    gpuTrainStepDevice(ctx, mode, features.data(), ctx.d_targets,
        flow ? ctx.d_z0 : nullptr, flow ? ctx.d_t : nullptr, 0.01f);
  });
  rejects<std::invalid_argument>([&] {
    gpuTrainStep(ctx, flow ? LossMode::MSE : LossMode::CICFM,
        features, target, flow ? nullptr : &z0, flow ? nullptr : &t, 0.01f);
  });
  if (flow) {
    const Tensor wrong_time({2, 1, 1});
    rejects<std::invalid_argument>([&] {
      gpuTrainStep(ctx, mode, features, target, z, &wrong_time, 0.01f);
    });
  }
  const float saved = features.data()[11];
  features.data()[11] = std::numeric_limits<float>::quiet_NaN();
  rejects<std::runtime_error>(step);
  features.data()[11] = saved;
  target.data()[3] = std::numeric_limits<float>::infinity();
  rejects<std::runtime_error>(step);
  target.data()[3] = 1e30f; // Finite inputs whose squared loss overflows.
  rejects<std::runtime_error>(step);
  target.data()[3] = 0.5f;
  if (flow) {
    t.data()[1] = 1.1f;
    rejects<std::runtime_error>(step);
    t.data()[1] = 0.5f;
    z0.data()[3] = std::numeric_limits<float>::quiet_NaN();
    rejects<std::runtime_error>(step);
    // Finite endpoints whose velocity overflows during assembly.
    z0.data()[3] = -std::numeric_limits<float>::max();
    target.data()[3] = std::numeric_limits<float>::max();
    rejects<std::runtime_error>(step);
  }
  matchesModel(ctx, model, true);
  freeGpuTrainContext(ctx);
}

void overflowingUpdate() {
  MlpConfig config{1, 1, 1, 3, ActivationKind::SiLU};
  Mlp model = createMlp(config, 1);
  for (auto &layer : model.layers) {
    layer.weight.data()[0] = 0;
    layer.bias.data()[0] = 0;
  }
  auto ctx = createGpuTrainContext(model, config, 1, 1, 1);
  Tensor features({1, 1}), target({1, 1}); target.data()[0] = 10;
  rejects<std::runtime_error>([&] {
    gpuTrainStep(ctx, LossMode::MSE, features, target, nullptr, nullptr,
                 std::numeric_limits<float>::max());
  });
  matchesModel(ctx, model, true);
  freeGpuTrainContext(ctx);
}

} // namespace

int main() {
  try {
    invalidMetadata();
    for (auto mode : {LossMode::MSE, LossMode::CICFM}) {
      parity(mode);
      invalidSteps(mode);
    }
    overflowingUpdate();
    std::cout << "test_train_step_device PASS\n";
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
