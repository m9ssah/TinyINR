#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

#include "model/mlp.h"

static void assertNear(float actual, float expected) {
  assert(std::fabs(actual - expected) < 1e-5f);
}

int main() {
  Tensor x({1, 2});
  x.at({0, 0}) = -1.0f;
  x.at({0, 1}) = 2.0f;

  Tensor y = silu(x);
  assertNear(y.at({0, 0}), -1.0f / (1.0f + std::exp(1.0f)));
  assertNear(y.at({0, 1}), 2.0f / (1.0f + std::exp(-2.0f)));

  MlpConfig config;
  config.input_dim = 4;
  config.hidden_width = 5;
  config.output_dim = 3;
  config.hidden_layer_count = 3;
  config.activation = ActivationKind::SiLU;

  Mlp model = createMlp(config, 123);
  Tensor input({2, 7, 4});
  for (int64_t i = 0; i < input.numel(); ++i) {
    input.data()[static_cast<size_t>(i)] = 0.01f * static_cast<float>(i);
  }

  MlpForwardResult result = mlpForward(model, input);
  assert(result.output.shape() == std::vector<int64_t>({2, 7, 3}));
  assert(result.cache.input.shape() == std::vector<int64_t>({14, 4}));
  assert(result.cache.pre_activations.size() == 3);
  assert(result.cache.activations.size() == 3);

  Mlp same_seed = createMlp(config, 123);
  MlpForwardResult repeated = mlpForward(same_seed, input);
  for (int64_t i = 0; i < result.output.numel(); ++i) {
    assertNear(result.output.data()[static_cast<size_t>(i)],
               repeated.output.data()[static_cast<size_t>(i)]);
  }

  std::cout << "test_mlp_forward PASS\n";
  return 0;
}
