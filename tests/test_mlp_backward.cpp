#include <cassert>
#include <cmath>
#include <iostream>

#include "model/mlp.h"
#include "training/loss.h"
#include "training/train_debug.h"

static float lossFor(Mlp &model, const Tensor &input, const Tensor &target) {
  MlpForwardResult result = mlpForward(model, input);
  return mseLoss(result.output, target);
}

static void assertClose(float actual, float expected) {
  const float abs_error = std::fabs(actual - expected);
  const float rel_error =
      abs_error / std::fmax(1e-8f, std::fabs(actual) + std::fabs(expected));
  assert(abs_error < 1e-3f || rel_error < 1e-2f);
}

int main() {
  MlpConfig config;
  config.input_dim = 5;
  config.hidden_width = 4;
  config.output_dim = 3;
  config.hidden_layer_count = 3;
  config.activation = ActivationKind::SiLU;

  Mlp model = createMlp(config, 7);
  Tensor input({1, 2, 5});
  Tensor target({1, 2, 3});
  for (int64_t i = 0; i < input.numel(); ++i) {
    input.data()[static_cast<size_t>(i)] = -0.2f + 0.05f * static_cast<float>(i);
  }
  for (int64_t i = 0; i < target.numel(); ++i) {
    target.data()[static_cast<size_t>(i)] =
        0.1f * std::sin(static_cast<float>(i + 1));
  }

  zeroGrad(model);
  MlpForwardResult result = mlpForward(model, input);
  Tensor output_gradient = mseOutputGradient(result.output, target);
  Tensor input_gradient = mlpBackward(model, result.cache, output_gradient);
  assert(input_gradient.shape() == std::vector<int64_t>({1, 2, 5}));
  assert(allGradientsFinite(model));

  const float eps = 1e-3f;

  float &w0 = model.layers[0].weight.at({0, 0});
  const float original_w0 = w0;
  w0 = original_w0 + eps;
  const float loss_plus_w0 = lossFor(model, input, target);
  w0 = original_w0 - eps;
  const float loss_minus_w0 = lossFor(model, input, target);
  w0 = original_w0;
  const float numeric_w0 = (loss_plus_w0 - loss_minus_w0) / (2.0f * eps);
  assertClose(model.layers[0].grad_weight.at({0, 0}), numeric_w0);

  float &b3 = model.layers[3].bias.at({1});
  const float original_b3 = b3;
  b3 = original_b3 + eps;
  const float loss_plus_b3 = lossFor(model, input, target);
  b3 = original_b3 - eps;
  const float loss_minus_b3 = lossFor(model, input, target);
  b3 = original_b3;
  const float numeric_b3 = (loss_plus_b3 - loss_minus_b3) / (2.0f * eps);
  assertClose(model.layers[3].grad_bias.at({1}), numeric_b3);

  std::cout << "test_mlp_backward PASS\n";
  return 0;
}
