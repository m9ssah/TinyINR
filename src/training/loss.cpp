#include "training/loss.h"

#include <stdexcept>

namespace {

void requireSameShape(const Tensor &output, const Tensor &target) {
  if (output.shape() != target.shape()) {
    throw std::invalid_argument("loss tensors must have the same shape");
  }
}

} // namespace

float mseLoss(const Tensor &output, const Tensor &target) {
  requireSameShape(output, target);

  float sum = 0.0f;
  for (int64_t i = 0; i < output.numel(); ++i) {
    const float diff = output.data()[static_cast<size_t>(i)] -
                       target.data()[static_cast<size_t>(i)];
    sum += diff * diff;
  }
  return sum / static_cast<float>(output.numel());
}

Tensor mseOutputGradient(const Tensor &output, const Tensor &target) {
  requireSameShape(output, target);

  Tensor gradient(output.shape());
  const float scale = 2.0f / static_cast<float>(output.numel());
  for (int64_t i = 0; i < output.numel(); ++i) {
    gradient.data()[static_cast<size_t>(i)] =
        scale * (output.data()[static_cast<size_t>(i)] -
                 target.data()[static_cast<size_t>(i)]);
  }
  return gradient;
}

MseLossResult mseLossAndGradient(const Tensor &output, const Tensor &target) {
  return MseLossResult{mseLoss(output, target),
                       mseOutputGradient(output, target)};
}

int64_t mseInputDim(int64_t coord_dim, int64_t num_frequencies) {
  if (coord_dim <= 0 || num_frequencies <= 0) {
    throw std::invalid_argument(
        "coord_dim and num_frequencies must be positive");
  }
  return coord_dim + 2 * coord_dim * num_frequencies;
}

int64_t cicfmInputDim(int64_t coord_dim, int64_t value_dim,
                      int64_t num_frequencies) {
  if (value_dim <= 0) {
    throw std::invalid_argument("value_dim must be positive");
  }
  return mseInputDim(coord_dim, num_frequencies) + value_dim + 1;
}
