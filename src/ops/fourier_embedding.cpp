/*
 *Tensor Wrapper Version
 */

#include "fourier_embedding.h"
#include "fourier_embedding_raw.h"

Tensor FourierEmbedding(const Tensor &input, int num_frequencies) {
  int N = input.shape[0];
  int D = input.shape[1];

  int output_dim = D + 2 * D * num_frequencies;

  Tensor output({N, output_dim});
  rawFourierEmbedding(input.data(), output.data(), N, D, num_frequencies);
  return output;
}
