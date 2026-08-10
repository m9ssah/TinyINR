#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

#include "model/mlp.h"
#include "training/loss.h"
#include "training/train_step.h"

static void fillFullSizeFeatures(Tensor &features) {
  for (int64_t n = 0; n < features.shape()[1]; ++n) {
    for (int64_t d = 0; d < features.shape()[2]; ++d) {
      features.at({0, n, d}) =
          0.01f * std::sin(static_cast<float>((n + 1) * (d + 1)));
    }
  }
}

static void fillFullSizeTargets(Tensor &targets) {
  for (int64_t n = 0; n < targets.shape()[1]; ++n) {
    const float x = static_cast<float>(n) /
                    static_cast<float>(targets.shape()[1] - 1);
    targets.at({0, n, 0}) = 0.25f + 0.5f * x;
    targets.at({0, n, 1}) = 0.75f - 0.4f * x;
    targets.at({0, n, 2}) = 0.5f + 0.1f * std::sin(6.2831853f * x);
  }
}

static TrainStepConfig makeConfig(LossMode mode) {
  TrainStepConfig config;
  config.loss_mode = mode;
  config.coord_dim = 2;
  config.value_dim = 3;
  config.num_frequencies = 8;
  config.learning_rate = 0.001f;
  config.deterministic_cicfm = true;
  config.seed = 401;
  return config;
}

static void assertFiniteResult(const TrainStepResult &result,
                               const std::vector<int64_t> &input_shape) {
  assert(std::isfinite(result.loss));
  assert(result.model_input.shape() == input_shape);
  assert(result.model_output.shape() == std::vector<int64_t>({1, 16, 3}));
  assert(result.loss_target.shape() == std::vector<int64_t>({1, 16, 3}));
  assert(result.output_gradient.shape() == std::vector<int64_t>({1, 16, 3}));
  assert(result.output_stats.all_finite);
  assert(result.gradient_stats.all_finite);
}

int main() {
  const int64_t coord_dim = 2;
  const int64_t value_dim = 3;
  const int64_t frequencies = 8;
  const int64_t points = 16;

  Tensor features({1, points, mseInputDim(coord_dim, frequencies)});
  Tensor targets({1, points, value_dim});
  fillFullSizeFeatures(features);
  fillFullSizeTargets(targets);

  Mlp mse_model =
      createMlp(makeBaselineMlpConfig(mseInputDim(coord_dim, frequencies),
                                      value_dim),
                501);
  assert(mse_model.config.hidden_width == 256);
  assert(mse_model.layers.size() == 4);

  TrainStepResult mse =
      trainStep(mse_model, features, targets, makeConfig(LossMode::MSE));
  assertFiniteResult(mse, {1, points, 34});

  for (int step = 0; step < 5; ++step) {
    mse = trainStep(mse_model, features, targets, makeConfig(LossMode::MSE));
    assertFiniteResult(mse, {1, points, 34});
  }

  Mlp cicfm_model =
      createMlp(makeBaselineMlpConfig(
                    cicfmInputDim(coord_dim, value_dim, frequencies),
                    value_dim),
                601);
  assert(cicfm_model.config.hidden_width == 256);
  assert(cicfm_model.layers.size() == 4);

  TrainStepResult cicfm =
      trainStep(cicfm_model, features, targets, makeConfig(LossMode::CICFM));
  assertFiniteResult(cicfm, {1, points, 38});

  for (int step = 0; step < 5; ++step) {
    cicfm =
        trainStep(cicfm_model, features, targets, makeConfig(LossMode::CICFM));
    assertFiniteResult(cicfm, {1, points, 38});
  }

  std::cout << "test_full_size_baseline PASS\n";
  return 0;
}
