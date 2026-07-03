#include "fourier_embedding_raw.h"

void rawFourierEmbedding(const float *input, float *output, int N, int D,
                         int F) {
  int output_dim = D + 2 * D * F;

  for (int i = 0; i < N; i++) {
    const float *in_row = input + i * D;
    float *out_row = output + i * output_dim;

    int out_idx = 0;

    for (int d = 0; d < D; d++) {
      out_row[out_idx++] = in_row[d];
    }

    for (int f = 0; f < F; f++) {
      float freq = std::pow(2.0f, f) * M_PI;
      for (int d = 0; d < D; d++) {
        float x = in_row[d];
        out_row[out_idx++] = std::sin(freq * x);
        out_row[out_idx++] = std::cos(freq * x);
      }
    }
  }
}