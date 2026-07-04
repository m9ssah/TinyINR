#include "include/ops/coordinate_batch.h"

CoordinateBatch::CoordinateBatch(const Tensor &coordinates,
                                 const Tensor &values)
    : coordinates_(coordinates), values_(values) {}

const Tensor &CoordinateBatch::coordinates() const { return coordinates_; }

const Tensor &CoordinateBatch::values() const { return values_; }

int64_t CoordinateBatch::batch_size() const { return coordinates_.shape()[0]; }

int64_t CoordinateBatch::num_points() const { return coordinates_.shape()[1]; }

int64_t CoordinateBatch::coord_dim() const { return coordinates_.shape()[2]; }

int64_t CoordinateBatch::value_dim() const { return values_.shape()[2]; }
