#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

#include "ops/attention.h"

int main() {
  CoordinateBatch batch = generateCoordinateBatch({0.0f, 1.0f}, 1, 2, 1);
  Tensor latents({1, 2, 2});
  latents.at({0, 0, 0}) = 1.0f;
  latents.at({0, 0, 1}) = 2.0f;
  latents.at({0, 1, 0}) = 3.0f;
  latents.at({0, 1, 1}) = 4.0f;

  AttentionConfig config{2, 2, 1, 2};
  AttentionWeights weights{Tensor({2, 2}), Tensor({2}),    Tensor({2, 4}),
                           Tensor({4}),    Tensor({2, 2}), Tensor({2})};
  // 0 queries give uniform attention
  // the value projection is identity
  weights.w_KV.at({0, 2}) = 1.0f;
  weights.w_KV.at({0, 3}) = 1.0f;
  weights.w_KV.at({1, 2}) = 1.0f;
  weights.w_KV.at({1, 3}) = 1.0f;
  weights.w_Q.at({0, 0}) = 1.0f;
  weights.w_Q.at({1, 1}) = 1.0f;

  Tensor output =
      latent_to_coordinate_attention(weights, batch, latents, config);
  assert(output.shape() == std::vector<int64_t>({1, 2, 2}));
  for (int64_t n = 0; n < 2; ++n) {
    assert(std::fabs(output.at({0, n, 0}) - 2.0f) < 1e-6f);
    assert(std::fabs(output.at({0, n, 1}) - 3.0f) < 1e-6f);
  }

  config.input_dim = 3;
  bool threw = false;
  try {
    (void)latent_to_coordinate_attention(weights, batch, latents, config);
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  assert(threw);
  std::cout << "test_attention PASS\n";
}
