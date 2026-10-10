#pragma once

#include <cstdint>
#include <vector>

#include "../tensor.h"

/*
goals:
- image/grid coordinate generator
- coordinate-value batch abstraction
- normalized coordinates, ideally [−1,1]
- uniform random subsampling of coordinates
*/

class CoordinateBatch {
private:
  Tensor coordinates_;
  Tensor values_;

public:
  CoordinateBatch(const Tensor &coordinates, const Tensor &values);

  const Tensor &coordinates() const;
  const Tensor &values() const;

  int64_t batch_size() const;
  int64_t num_points() const;
  int64_t coord_dim() const;
  int64_t value_dim() const;
};

CoordinateBatch generateCoordinateBatch(const std::vector<float> &image,
                                        int64_t height, int64_t width,
                                        int64_t channels);

CoordinateBatch randomSubsample(const CoordinateBatch &batch,
                                int64_t sample_count);

// Copies K points per example in the requested order, preserving duplicates.
// indices[b*K+k] is an original point index local to example b. Requires
// 1 <= K <= N and exactly B*K indices in [0,N); invalid inputs throw
// std::invalid_argument. The source batch and indices are unchanged.
CoordinateBatch gatherCoordinateBatch(const CoordinateBatch &batch,
                                     const std::vector<int64_t> &indices,
                                     int64_t sample_count);
