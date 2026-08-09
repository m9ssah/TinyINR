#pragma once

#include "../tensor.h"

enum class LossMode { MSE, CICFM };

struct MseLossResult {
  float loss;
  Tensor output_gradient;
};

float mseLoss(const Tensor &output, const Tensor &target);
Tensor mseOutputGradient(const Tensor &output, const Tensor &target);
MseLossResult mseLossAndGradient(const Tensor &output, const Tensor &target);

int64_t mseInputDim(int64_t coord_dim, int64_t num_frequencies);
int64_t cicfmInputDim(int64_t coord_dim, int64_t value_dim,
                      int64_t num_frequencies);
