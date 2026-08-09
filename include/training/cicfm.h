#pragma once

#include <cstdint>

#include "../tensor.h"

struct CicfmBatch {
  Tensor z0;
  Tensor z1;
  Tensor t;
  Tensor zt;
  Tensor target_velocity;
  Tensor input;
};

Tensor sampleUniform(const std::vector<int64_t> &shape, float low, float high,
                     uint32_t seed);
Tensor concatFeaturesZtT(const Tensor &features, const Tensor &zt,
                         const Tensor &t);
CicfmBatch makeCicfmBatch(const Tensor &features, const Tensor &targets,
                          const Tensor &z0, const Tensor &t);
CicfmBatch makeDeterministicCicfmBatch(const Tensor &features,
                                       const Tensor &targets);
