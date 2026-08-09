#pragma once

#include <cstdint>
#include <vector>

#include "../tensor.h"

enum class ActivationKind { SiLU };

struct MlpConfig {
  int64_t input_dim;
  int64_t hidden_width;
  int64_t output_dim;
  int64_t hidden_layer_count;
  ActivationKind activation;
};

struct LinearLayer {
  Tensor weight;
  Tensor bias;
  Tensor grad_weight;
  Tensor grad_bias;

  LinearLayer(int64_t in_dim, int64_t out_dim);
};

struct Mlp {
  MlpConfig config;
  std::vector<LinearLayer> layers;

  explicit Mlp(const MlpConfig &config);
};

struct MlpForwardCache {
  Tensor input;
  std::vector<Tensor> pre_activations;
  std::vector<Tensor> activations;
  std::vector<int64_t> input_shape;
  std::vector<int64_t> output_shape;
};

struct MlpForwardResult {
  Tensor output;
  MlpForwardCache cache;
};

MlpConfig makeBaselineMlpConfig(int64_t input_dim, int64_t output_dim);
Mlp createMlp(const MlpConfig &config, uint32_t seed);

Tensor silu(const Tensor &input);
Tensor siluBackward(const Tensor &pre_activation, const Tensor &upstream);

MlpForwardResult mlpForward(const Mlp &model, const Tensor &input);
Tensor mlpBackward(Mlp &model, const MlpForwardCache &cache,
                   const Tensor &output_gradient);

void zeroGrad(Mlp &model);
void sgdStep(Mlp &model, float learning_rate);
