#include "training/train_debug.h"

#include <cmath>

TensorStats tensorStats(const Tensor &tensor) {
  TensorStats stats;
  stats.all_finite = true;
  stats.mean_abs = 0.0f;
  stats.max_abs = 0.0f;

  for (int64_t i = 0; i < tensor.numel(); ++i) {
    const float value = tensor.data()[static_cast<size_t>(i)];
    if (!std::isfinite(value)) {
      stats.all_finite = false;
    }
    const float abs_value = std::fabs(value);
    stats.mean_abs += abs_value;
    if (abs_value > stats.max_abs) {
      stats.max_abs = abs_value;
    }
  }

  stats.mean_abs /= static_cast<float>(tensor.numel());
  return stats;
}

bool allParametersFinite(const Mlp &model) {
  for (size_t layer_index = 0; layer_index < model.layers.size();
       ++layer_index) {
    const LinearLayer &layer = model.layers[layer_index];
    if (!tensorStats(layer.weight).all_finite ||
        !tensorStats(layer.bias).all_finite) {
      return false;
    }
  }
  return true;
}

bool allGradientsFinite(const Mlp &model) {
  for (size_t layer_index = 0; layer_index < model.layers.size();
       ++layer_index) {
    const LinearLayer &layer = model.layers[layer_index];
    if (!tensorStats(layer.grad_weight).all_finite ||
        !tensorStats(layer.grad_bias).all_finite) {
      return false;
    }
  }
  return true;
}
