#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

#include "training/cicfm.h"
#include "training/loss.h"

static void assertNear(float actual, float expected) {
  assert(std::fabs(actual - expected) < 1e-6f);
}

int main() {
  Tensor features({1, 2, 34});
  Tensor targets({1, 2, 3});
  for (int64_t i = 0; i < features.numel(); ++i) {
    features.data()[static_cast<size_t>(i)] = 0.01f * static_cast<float>(i);
  }
  for (int64_t i = 0; i < targets.numel(); ++i) {
    targets.data()[static_cast<size_t>(i)] = 0.1f * static_cast<float>(i + 1);
  }

  CicfmBatch deterministic = makeDeterministicCicfmBatch(features, targets);
  assert(deterministic.input.shape() == std::vector<int64_t>({1, 2, 38}));

  for (int64_t i = 0; i < targets.numel(); ++i) {
    assertNear(deterministic.zt.data()[static_cast<size_t>(i)],
               0.5f * targets.data()[static_cast<size_t>(i)]);
    assertNear(deterministic.target_velocity.data()[static_cast<size_t>(i)],
               targets.data()[static_cast<size_t>(i)]);
  }
  assertNear(deterministic.t.at({0, 0, 0}), 0.5f);
  assertNear(deterministic.t.at({0, 1, 0}), 0.5f);

  Tensor predicted_velocity({1, 1, 3});
  Tensor z0({1, 1, 3});
  Tensor z1({1, 1, 3});
  Tensor t({1, 1, 1});
  z0.at({0, 0, 0}) = 0.2f;
  z0.at({0, 0, 1}) = 0.4f;
  z0.at({0, 0, 2}) = 0.6f;
  z1.at({0, 0, 0}) = 0.5f;
  z1.at({0, 0, 1}) = 0.1f;
  z1.at({0, 0, 2}) = 0.9f;
  t.at({0, 0, 0}) = 0.25f;

  Tensor one_feature({1, 1, 34});
  CicfmBatch batch = makeCicfmBatch(one_feature, z1, z0, t);
  assertNear(batch.target_velocity.at({0, 0, 0}), 0.3f);
  assertNear(batch.target_velocity.at({0, 0, 1}), -0.3f);
  assertNear(batch.target_velocity.at({0, 0, 2}), 0.3f);
  assertNear(mseLoss(predicted_velocity, batch.target_velocity), 0.09f);

  std::cout << "test_cicfm PASS\n";
  return 0;
}
