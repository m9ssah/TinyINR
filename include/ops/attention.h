#pragma once

#include "ops/coordinate_batch.h"
#include "tensor.h"


/* shapes & math ref:
  Q        = CoordinateFeatures @ W_q                    [B, N, D_model]
  [K | V]  = Latents @ W_kv                              [B, L, 2 * D_model]
  scores   = (Q @ K^T) / sqrt(d_k)                       [B, H, N, L]
  attn     = softmax(scores, dim = -1)                   [B, H, N, L]
  output   = attn @ V                                    [B, N, D_model]

D_model is split evenly across H heads: d_k = d_v = D_model / H.
Output dim == D_model == input latent dim, by construction
this is what lets the block sit behind a residual connection later without a
reshape
*/

struct AttentionConfig {
  int64_t input_dim;
  int64_t latent_dim;  // D_model
  int64_t num_heads;   // H
  int64_t num_latents; // L
};

struct AttentionWeights {
  Tensor w_Q;  // [inpput_dim, latent_dim]
  Tensor b_Q;  // [latent_dim]
  Tensor w_KV; // [latent_dim, 2 * latent_dim]
  Tensor b_KV; // [2 * latent_dim]
};

struct AttentionInputs {
  Tensor features; // [B, N, input_dim] (query source)
  Tensor latents;  // [B, L, latent_dim] (kv source)
};

// generic cross-attention
Tensor attention(const AttentionWeights &weights, const AttentionInputs &inputs,
                 const AttentionConfig &config);

// cross-attention with CoordinateBatch's raw coordinates as queries instead of
// an embedded feature tensor. config.input_dim must equal batch.coord_dim()
Tensor latent_to_coordinate_attention(const AttentionWeights &weights,
                                      const CoordinateBatch &batch,
                                      const Tensor &latents,
                                      const AttentionConfig &config);
