#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
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

static void testLossOverflowRejected() {
  for (LossMode mode : {LossMode::MSE, LossMode::CICFM}) {
    TrainStepConfig config{mode, 2, 1, 1, 0.01f, true, 123};
    const int64_t input_dim = mode == LossMode::MSE ? 6 : 8;
    Mlp model(MlpConfig{input_dim, 1, 1, 3, ActivationKind::SiLU});
    const Mlp before = model;
    Tensor features({1, 1, 6});
    Tensor targets({1, 1, 1});
    targets.data()[0] = 1e20f;
    bool threw = false;
    try {
      (void)trainStep(model, features, targets, config);
    } catch (const std::runtime_error &) {
      threw = true;
    }
    assert(threw);
    for (size_t layer = 0; layer < model.layers.size(); ++layer) {
      for (int64_t i = 0; i < model.layers[layer].weight.numel(); ++i) {
        assert(model.layers[layer].weight.data()[i] ==
               before.layers[layer].weight.data()[i]);
      }
      for (int64_t i = 0; i < model.layers[layer].bias.numel(); ++i) {
        assert(model.layers[layer].bias.data()[i] ==
               before.layers[layer].bias.data()[i]);
      }
    }
  }
}

static void testInvalidLearningRate() {
  Tensor features({1, 1, 34});
  Tensor targets({1, 1, 3});
  for (float rate : {0.0f, -1.0f, std::numeric_limits<float>::infinity(),
                     std::numeric_limits<float>::quiet_NaN()}) {
    Mlp model(MlpConfig{34, 1, 3, 3, ActivationKind::SiLU});
    auto config = baseConfig(LossMode::MSE);
    config.learning_rate = rate;
    bool threw = false;
    try {
      (void)trainStep(model, features, targets, config);
    } catch (const std::invalid_argument &) {
      threw = true;
    }
    assert(threw);
    assert(allParametersFinite(model));
  }
}

int main() {
  testLossOverflowRejected();
  testInvalidLearningRate();
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
