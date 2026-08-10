#include <cassert>
#include <cmath>
#include <iostream>

#include "training/loss.h"

static void assertNear(float actual, float expected) {
  assert(std::fabs(actual - expected) < 1e-6f);
}

int main() {
  Tensor output({1, 1, 3});
  Tensor target({1, 1, 3});
  output.at({0, 0, 0}) = 0.0f;
  output.at({0, 0, 1}) = 0.0f;
  output.at({0, 0, 2}) = 0.0f;
  target.at({0, 0, 0}) = 0.3f;
  target.at({0, 0, 1}) = -0.3f;
  target.at({0, 0, 2}) = 0.3f;

  assertNear(mseLoss(output, target), 0.09f);
  Tensor gradient = mseOutputGradient(output, target);
  assertNear(gradient.at({0, 0, 0}), -0.2f);
  assertNear(gradient.at({0, 0, 1}), 0.2f);
  assertNear(gradient.at({0, 0, 2}), -0.2f);

  assert(mseInputDim(2, 8) == 34);
  assert(mseInputDim(2, 16) == 66);
  assert(cicfmInputDim(2, 3, 8) == 38);
  assert(cicfmInputDim(2, 3, 16) == 70);

  std::cout << "test_loss PASS\n";
  return 0;
}
