#include <cassert>
#include <cmath>
#include <iostream>

#include "model/mlp.h"
#include "training/cicfm.h"
#include "training/loss.h"
#include "training/train_debug.h"

static float trainStep(Mlp &model, const Tensor &input, const Tensor &target,
                       float learning_rate) {
  zeroGrad(model);
  MlpForwardResult result = mlpForward(model, input);
  MseLossResult loss = mseLossAndGradient(result.output, target);
  assert(std::isfinite(loss.loss));
  assert(tensorStats(result.output).all_finite);
  assert(tensorStats(loss.output_gradient).all_finite);
  (void)mlpBackward(model, result.cache, loss.output_gradient);
  assert(allGradientsFinite(model));
  sgdStep(model, learning_rate);
  assert(allParametersFinite(model));
  return loss.loss;
}

static void fillSyntheticTarget(Tensor &target) {
  for (int64_t b = 0; b < target.shape()[0]; ++b) {
    for (int64_t n = 0; n < target.shape()[1]; ++n) {
      target.at({b, n, 0}) = 0.2f + 0.03f * static_cast<float>(n);
      target.at({b, n, 1}) = 0.4f - 0.02f * static_cast<float>(n);
      target.at({b, n, 2}) = 0.1f * std::sin(static_cast<float>(n));
    }
  }
}

int main() {
  MlpConfig mse_config;
  mse_config.input_dim = 5;
  mse_config.hidden_width = 8;
  mse_config.output_dim = 3;
  mse_config.hidden_layer_count = 3;
  mse_config.activation = ActivationKind::SiLU;

  Tensor features({1, 16, 5});
  Tensor targets({1, 16, 3});
  for (int64_t i = 0; i < features.numel(); ++i) {
    features.data()[static_cast<size_t>(i)] =
        -0.5f + 0.01f * static_cast<float>(i);
  }
  fillSyntheticTarget(targets);

  Mlp mse_model = createMlp(mse_config, 11);
  const float mse_initial = trainStep(mse_model, features, targets, 0.03f);
  float mse_final = mse_initial;
  for (int step = 0; step < 200; ++step) {
    mse_final = trainStep(mse_model, features, targets, 0.03f);
  }
  assert(mse_final < mse_initial);

  CicfmBatch deterministic = makeDeterministicCicfmBatch(features, targets);
  MlpConfig cicfm_config = mse_config;
  cicfm_config.input_dim = deterministic.input.shape()[2];
  Mlp cicfm_model = createMlp(cicfm_config, 13);

  const float cicfm_initial =
      trainStep(cicfm_model, deterministic.input,
                deterministic.target_velocity, 0.03f);
  float cicfm_final = cicfm_initial;
  for (int step = 0; step < 200; ++step) {
    cicfm_final =
        trainStep(cicfm_model, deterministic.input,
                  deterministic.target_velocity, 0.03f);
  }
  assert(cicfm_final < cicfm_initial);

  Tensor random_z0 = sampleUniform(targets.shape(), 0.0f, 1.0f, 101);
  Tensor random_t =
      sampleUniform({targets.shape()[0], targets.shape()[1], 1}, 0.0f, 1.0f,
                    102);
  CicfmBatch random_batch =
      makeCicfmBatch(features, targets, random_z0, random_t);
  const float random_loss =
      trainStep(cicfm_model, random_batch.input, random_batch.target_velocity,
                0.01f);
  assert(std::isfinite(random_loss));

  std::cout << "test_training_debug PASS\n";
  return 0;
}
