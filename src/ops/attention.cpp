#include "ops/attention.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require_shape(const Tensor &tensor, const std::vector<int64_t> &shape,
                   const char *name) {
  if (tensor.shape() != shape) {
    throw std::invalid_argument(std::string(name) + " has an invalid shape");
  }
}

void validate_attention_arguments(const AttentionWeights &weights,
                                  const AttentionInputs &inputs,
                                  const AttentionConfig &config) {
  if (config.input_dim <= 0 || config.latent_dim <= 0 ||
      config.num_heads <= 0 || config.num_latents <= 0 ||
      config.latent_dim % config.num_heads != 0) {
    throw std::invalid_argument("invalid attention configuration");
  }
  if (inputs.features.ndim() != 3 || inputs.latents.ndim() != 3) {
    throw std::invalid_argument("attention inputs must have shape [B, N, D]");
  }

  const int64_t batch_size = inputs.features.shape()[0];
  if (inputs.features.shape()[2] != config.input_dim ||
      inputs.latents.shape() !=
          std::vector<int64_t>(
              {batch_size, config.num_latents, config.latent_dim})) {
    throw std::invalid_argument("attention input shapes do not match config");
  }

  require_shape(weights.w_Q, {config.input_dim, config.latent_dim}, "w_Q");
  require_shape(weights.b_Q, {config.latent_dim}, "b_Q");
  require_shape(weights.w_KV, {config.latent_dim, 2 * config.latent_dim},
                "w_KV");
  require_shape(weights.b_KV, {2 * config.latent_dim}, "b_KV");
}

} // namespace

Tensor attention(const AttentionWeights &weights, const AttentionInputs &inputs,
                 const AttentionConfig &config) {};

Tensor latent_to_coordinate_attention(const AttentionWeights &weights,
                                      const CoordinateBatch &batch,
                                      const Tensor &latents,
                                      const AttentionConfig &config) {
  return attention(weights, {batch.coordinates(), latents}, config);
};
