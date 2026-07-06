#pragma once

#include <cstdint>
#include <vector>

#include "tensor.h"

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
