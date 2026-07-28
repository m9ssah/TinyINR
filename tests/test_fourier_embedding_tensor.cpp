#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>

#include "ops/fourier_embedding.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static float expected(float x, int f, bool cosine) {
  float angle =
      std::pow(2.0f, static_cast<float>(f)) * static_cast<float>(M_PI) * x;
  return cosine ? std::cos(angle) : std::sin(angle);
}

int main() {
  Tensor input({2, 2});
  input.at({0, 0}) = 0.0f;
  input.at({0, 1}) = 0.5f;
  input.at({1, 0}) = -1.0f;
  input.at({1, 1}) = 1.0f;

  const int F = 2;
  Tensor output = fourierEmbedding(input, F);

  assert(output.shape() == std::vector<int64_t>({2, 10}));

  for (int n = 0; n < 2; n++) {
    int out_idx = 0;
    for (int d = 0; d < 2; d++) {
      assert(std::fabs(output.at({n, out_idx}) - input.at({n, d})) < 1e-5f);
      out_idx++;
    }

    for (int f = 0; f < F; f++) {
      for (int d = 0; d < 2; d++) {
        float x = input.at({n, d});
        assert(std::fabs(output.at({n, out_idx}) - expected(x, f, false)) <
               1e-5f);
        out_idx++;
        assert(std::fabs(output.at({n, out_idx}) - expected(x, f, true)) <
               1e-5f);
        out_idx++;
      }
    }
  }

  bool threw = false;
  try {
    Tensor invalid({1, 2, 3});
    (void)fourierEmbedding(invalid, F);
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  assert(threw);

  std::cout << "test_fourier_embedding_tensor PASS\n";
  return 0;
}
