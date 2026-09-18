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
                 const AttentionConfig &config) {
  validate_attention_arguments(weights, inputs, config);

  const int64_t batch_size = inputs.features.shape()[0];         // B
  const int64_t num_points = inputs.features.shape()[1];         // N
  const int64_t head_dim = config.latent_dim / config.num_heads; // d_k

  Tensor queries(
      {batch_size, num_points, config.latent_dim}); // Q, [B, N, D_model]
  Tensor keys_values({batch_size, config.num_latents,
                      2 * config.latent_dim}); // [K | V], [B, L, 2 * D_model]
  Tensor output({batch_size, num_points, config.latent_dim}); // output

  for (int64_t b = 0; b < batch_size; b++) {
    // Q projection
    for (int64_t n = 0; n < num_points; n++) {
      for (int64_t d = 0; d < config.latent_dim; d++) {
        float value = weights.b_Q.at({d});
        for (int64_t i = 0; i < config.input_dim; i++) {
          value += inputs.features.at({b, n, i}) * weights.w_Q.at({i, d});
        }
        queries.at({b, n, d}) = value;
      }
    }
    // fused KV projection
    for (int64_t l = 0; l < config.num_latents; l++) {
      for (int64_t d = 0; d < 2 * config.latent_dim; d++) {
        float value = weights.b_KV.at({d});
        for (int64_t i = 0; i < config.latent_dim; i++) {
          value += inputs.latents.at({b, l, i}) * weights.w_KV.at({i, d});
        }
        keys_values.at({b, l, d}) = value;
      }
    }
  }
  const float scale =
      1.0f / std::sqrt(static_cast<float>(head_dim)); // 1/sqrt(d_k)
  for (int64_t b = 0; b < batch_size; b++) {
    for (int64_t h = 0; h < config.num_heads; h++) {
      const int64_t head_offset = h * head_dim;
      const int64_t key_offset = head_offset;
      const int64_t value_offset = config.latent_dim + head_offset;
      for (int64_t n = 0; n < num_points; n++) {
        std::vector<float> scores(static_cast<size_t>(config.num_latents));
        float max_score = -std::numeric_limits<float>::infinity();
        for (int64_t l = 0; l < config.num_latents; ++l) {
          float score = 0.0f;
          for (int64_t d = 0; d < head_dim; ++d) {
            score += queries.at({b, n, key_offset + d}) *
                     keys_values.at({b, l, key_offset + d});
          }
          scores[static_cast<size_t>(l)] = score * scale;
          max_score = std::max(max_score, score * scale);
        }
        float normalizer = 0.0f;
        for (float &score : scores) {
          score = std::exp(score - max_score);
          normalizer += score;
        }
        for (int64_t d = 0; d < head_dim; ++d) {
          float value = 0.0f;
          for (int64_t l = 0; l < config.num_latents; ++l) {
            value += scores[static_cast<size_t>(l)] / normalizer *
                     keys_values.at({b, l, value_offset + d});
          }
          output.at({b, n, key_offset + d}) = value;
        }
      }
    }
  }
  return output;
};

Tensor latent_to_coordinate_attention(const AttentionWeights &weights,
                                      const CoordinateBatch &batch,
                                      const Tensor &latents,
                                      const AttentionConfig &config) {
  return attention(weights, {batch.coordinates(), latents}, config);
};
