#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "model/mlp.h"
#include "training/loss.h"
#include "training/train_step.h"

static void fillFeatures(Tensor &features) {
  for (int64_t i = 0; i < features.numel(); ++i) {
    features.data()[static_cast<size_t>(i)] = 0.01f * static_cast<float>(i);
  }
}

static void fillTargets(Tensor &targets) {
  for (int64_t n = 0; n < targets.shape()[1]; ++n) {
    targets.at({0, n, 0}) = 0.2f + 0.05f * static_cast<float>(n);
    targets.at({0, n, 1}) = 0.3f - 0.02f * static_cast<float>(n);
    targets.at({0, n, 2}) = 0.1f * std::sin(static_cast<float>(n));
  }
}

static TrainStepConfig baseConfig(LossMode mode) {
  TrainStepConfig config;
  config.loss_mode = mode;
  config.coord_dim = 2;
  config.value_dim = 3;
  config.num_frequencies = 8;
  config.learning_rate = 0.01f;
  config.deterministic_cicfm = true;
  config.seed = 123;
  return config;
}

int main() {
  Tensor features({1, 4, mseInputDim(2, 8)});
  Tensor targets({1, 4, 3});
  fillFeatures(features);
  fillTargets(targets);

  MlpConfig mse_config;
  mse_config.input_dim = mseInputDim(2, 8);
  mse_config.hidden_width = 8;
  mse_config.output_dim = 3;
  mse_config.hidden_layer_count = 3;
  mse_config.activation = ActivationKind::SiLU;
  Mlp mse_model = createMlp(mse_config, 17);

  TrainStepResult mse =
      trainStep(mse_model, features, targets, baseConfig(LossMode::MSE));
  assert(mse.model_input.shape() == features.shape());
  assert(mse.loss_target.shape() == targets.shape());
  for (int64_t i = 0; i < targets.numel(); ++i) {
    assert(std::fabs(mse.loss_target.data()[static_cast<size_t>(i)] -
                     targets.data()[static_cast<size_t>(i)]) < 1e-6f);
  }

  MlpConfig cicfm_config = mse_config;
  cicfm_config.input_dim = cicfmInputDim(2, 3, 8);
  Mlp cicfm_model = createMlp(cicfm_config, 19);
  TrainStepResult cicfm =
      trainStep(cicfm_model, features, targets, baseConfig(LossMode::CICFM));
  assert(cicfm.model_input.shape() == std::vector<int64_t>({1, 4, 38}));
  assert(cicfm.loss_target.shape() == targets.shape());
  for (int64_t i = 0; i < targets.numel(); ++i) {
    assert(std::fabs(cicfm.loss_target.data()[static_cast<size_t>(i)] -
                     targets.data()[static_cast<size_t>(i)]) < 1e-6f);
  }

  TrainStepConfig random_cicfm_config = baseConfig(LossMode::CICFM);
  random_cicfm_config.deterministic_cicfm = false;
  random_cicfm_config.seed = 211;
  Tensor expected_z0 = sampleUniform(targets.shape(), 0.0f, 1.0f,
                                     random_cicfm_config.seed);
  Tensor expected_t = sampleUniform({1, 4, 1}, 0.0f, 1.0f,
                                    random_cicfm_config.seed + 1);
  CicfmBatch expected_batch =
      makeCicfmBatch(features, targets, expected_z0, expected_t);
  Mlp random_cicfm_model = createMlp(cicfm_config, 31);
  TrainStepResult random_cicfm =
      trainStep(random_cicfm_model, features, targets, random_cicfm_config);
  bool differs_from_targets = false;
  for (int64_t i = 0; i < targets.numel(); ++i) {
    const float actual = random_cicfm.loss_target.data()[static_cast<size_t>(i)];
    const float expected =
        expected_batch.target_velocity.data()[static_cast<size_t>(i)];
    assert(std::fabs(actual - expected) < 1e-6f);
    if (std::fabs(actual - targets.data()[static_cast<size_t>(i)]) > 1e-5f) {
      differs_from_targets = true;
    }
  }
  assert(differs_from_targets);

  bool threw = false;
  try {
    Mlp wrong_model = createMlp(mse_config, 23);
    (void)trainStep(wrong_model, features, targets, baseConfig(LossMode::CICFM));
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  assert(threw);

  threw = false;
  try {
    Mlp wrong_model = createMlp(cicfm_config, 29);
    (void)trainStep(wrong_model, features, targets, baseConfig(LossMode::MSE));
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  assert(threw);

  std::cout << "test_train_step PASS\n";
  return 0;
}
