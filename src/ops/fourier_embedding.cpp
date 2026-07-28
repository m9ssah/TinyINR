#include "ops/fourier_embedding.h"

#include <stdexcept>

#include "ops/fourier_embedding_raw.h"

Tensor fourierEmbedding(const Tensor &input, int num_frequencies) {
  if (input.ndim() != 2) {
    throw std::invalid_argument("fourierEmbedding expects input shape [N, D]");
  }
  if (num_frequencies <= 0) {
    throw std::invalid_argument("num_frequencies must be positive");
  }

  int N = static_cast<int>(input.shape()[0]);
  int D = static_cast<int>(input.shape()[1]);

  int output_dim = D + 2 * D * num_frequencies;

  Tensor output({N, output_dim});
  rawFourierEmbedding(input.data(), output.data(), N, D, num_frequencies);
  return output;
}
