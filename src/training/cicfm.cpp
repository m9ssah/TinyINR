#include "training/cicfm.h"

#include <random>
#include <stdexcept>
#include <string>

namespace {

void require3d(const Tensor &tensor, const char *name) {
  if (tensor.ndim() != 3) {
    throw std::invalid_argument(std::string(name) + " must be 3D");
  }
}

void requireSameBatchPoints(const Tensor &features, const Tensor &targets) {
  require3d(features, "features");
  require3d(targets, "targets");
  if (features.shape()[0] != targets.shape()[0] ||
      features.shape()[1] != targets.shape()[1]) {
    throw std::invalid_argument(
        "features and targets must share batch and point dimensions");
  }
}

} // namespace

Tensor sampleUniform(const std::vector<int64_t> &shape, float low, float high,
                     uint32_t seed) {
  if (low > high) {
    throw std::invalid_argument("uniform low must be <= high");
  }

  Tensor output(shape);
  std::mt19937 generator(seed);
  std::uniform_real_distribution<float> distribution(low, high);
  for (int64_t i = 0; i < output.numel(); ++i) {
    output.data()[static_cast<size_t>(i)] = distribution(generator);
  }
  return output;
}

Tensor concatFeaturesZtT(const Tensor &features, const Tensor &zt,
                         const Tensor &t) {
  requireSameBatchPoints(features, zt);
  require3d(t, "t");

  const int64_t batch = features.shape()[0];
  const int64_t points = features.shape()[1];
  const int64_t feature_dim = features.shape()[2];
  const int64_t channels = zt.shape()[2];

  if (t.shape() != std::vector<int64_t>({batch, points, 1})) {
    throw std::invalid_argument("t must have shape [B, N, 1]");
  }

  Tensor input({batch, points, feature_dim + channels + 1});
  for (int64_t b = 0; b < batch; ++b) {
    for (int64_t n = 0; n < points; ++n) {
      int64_t out = 0;
      for (int64_t f = 0; f < feature_dim; ++f) {
        input.at({b, n, out}) = features.at({b, n, f});
        ++out;
      }
      for (int64_t c = 0; c < channels; ++c) {
        input.at({b, n, out}) = zt.at({b, n, c});
        ++out;
      }
      input.at({b, n, out}) = t.at({b, n, 0});
    }
  }

  return input;
}

CicfmBatch makeCicfmBatch(const Tensor &features, const Tensor &targets,
                          const Tensor &z0, const Tensor &t) {
  requireSameBatchPoints(features, targets);
  if (z0.shape() != targets.shape()) {
    throw std::invalid_argument("z0 must match targets shape");
  }

  const int64_t batch = targets.shape()[0];
  const int64_t points = targets.shape()[1];
  const int64_t channels = targets.shape()[2];
  if (t.shape() != std::vector<int64_t>({batch, points, 1})) {
    throw std::invalid_argument("t must have shape [B, N, 1]");
  }

  Tensor z1 = targets;
  Tensor zt(targets.shape());
  Tensor target_velocity(targets.shape());

  for (int64_t b = 0; b < batch; ++b) {
    for (int64_t n = 0; n < points; ++n) {
      const float time = t.at({b, n, 0});
      if (time < 0.0f || time > 1.0f) {
        throw std::invalid_argument("t values must be in [0, 1]");
      }
      for (int64_t c = 0; c < channels; ++c) {
        const float source = z0.at({b, n, c});
        const float target = z1.at({b, n, c});
        zt.at({b, n, c}) = (1.0f - time) * source + time * target;
        target_velocity.at({b, n, c}) = target - source;
      }
    }
  }

  Tensor input = concatFeaturesZtT(features, zt, t);
  return CicfmBatch{z0, z1, t, zt, target_velocity, input};
}

CicfmBatch makeDeterministicCicfmBatch(const Tensor &features,
                                       const Tensor &targets) {
  Tensor z0 = Tensor::zeros(targets.shape());
  Tensor t({targets.shape()[0], targets.shape()[1], 1});
  for (int64_t i = 0; i < t.numel(); ++i) {
    t.data()[static_cast<size_t>(i)] = 0.5f;
  }

  return makeCicfmBatch(features, targets, z0, t);
}
