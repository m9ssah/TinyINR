#pragma once

#include "../model/mlp.h"
#include "../tensor.h"

struct TensorStats {
  bool all_finite;
  float mean_abs;
  float max_abs;
};

TensorStats tensorStats(const Tensor &tensor);
bool allParametersFinite(const Mlp &model);
bool allGradientsFinite(const Mlp &model);
