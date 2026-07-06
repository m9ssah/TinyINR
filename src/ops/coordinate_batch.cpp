#include <algorithm>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

#include "ops/coordinate_batch.h"

CoordinateBatch::CoordinateBatch(const Tensor &coordinates,
                                 const Tensor &values)
    : coordinates_(coordinates), values_(values) {
  if (coordinates_.ndim() != 3) {
    throw std::invalid_argument("coordinates must have shape [B, N, C]");
  }
  if (values_.ndim() != 3) {
    throw std::invalid_argument("values must have shape [B, N, V]");
  }
  if (coordinates_.shape()[0] != values_.shape()[0]) {
    throw std::invalid_argument(
        "coordinates and values must have the same batch size");
  }
  if (coordinates_.shape()[1] != values_.shape()[1]) {
    throw std::invalid_argument(
        "coordinates and values must have the same point count");
  }
}

const Tensor &CoordinateBatch::coordinates() const { return coordinates_; }

const Tensor &CoordinateBatch::values() const { return values_; }

int64_t CoordinateBatch::batch_size() const { return coordinates_.shape()[0]; }

int64_t CoordinateBatch::num_points() const { return coordinates_.shape()[1]; }

int64_t CoordinateBatch::coord_dim() const { return coordinates_.shape()[2]; }

int64_t CoordinateBatch::value_dim() const { return values_.shape()[2]; }

CoordinateBatch generateCoordinateBatch(const std::vector<float> &image,
                                        int64_t height, int64_t width,
                                        int64_t channels) {
  if (height <= 0 || width <= 0 || channels <= 0) {
    throw std::invalid_argument("height, width, and channels must be positive");
  }

  const int64_t num_points = height * width;
  if (static_cast<int64_t>(image.size()) != num_points * channels) {
    throw std::invalid_argument(
        "image size must equal height * width * channels");
  }

  Tensor coordinates({1, num_points, 2});
  Tensor values({1, num_points, channels});

  for (int64_t row = 0; row < height; row++) {
    for (int64_t col = 0; col < width; col++) {
      const int64_t point = row * width + col;
      const float x = width == 1 ? 0.0f : 2.0f * col / (width - 1) - 1.0f;
      const float y = height == 1 ? 0.0f : 2.0f * row / (height - 1) - 1.0f;

      coordinates.at({0, point, 0}) = x;
      coordinates.at({0, point, 1}) = y;

      for (int64_t channel = 0; channel < channels; channel++) {
        values.at({0, point, channel}) =
            image[static_cast<size_t>(point * channels + channel)];
      }
    }
  }

  return CoordinateBatch(coordinates, values);
}

CoordinateBatch randomSubsample(const CoordinateBatch &batch,
                                int64_t sample_count) {
  if (sample_count <= 0) {
    throw std::invalid_argument("sample_count must be positive");
  }
  if (sample_count > batch.num_points()) {
    throw std::invalid_argument("sample_count cannot exceed number of points");
  }

  Tensor coordinates({batch.batch_size(), sample_count, batch.coord_dim()});
  Tensor values({batch.batch_size(), sample_count, batch.value_dim()});

  std::random_device seed;
  std::mt19937 generator(seed());

  for (int64_t b = 0; b < batch.batch_size(); b++) {
    std::vector<int64_t> indices(static_cast<size_t>(batch.num_points()));
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), generator);

    for (int64_t sample = 0; sample < sample_count; sample++) {
      const int64_t source_point = indices[static_cast<size_t>(sample)];

      for (int64_t c = 0; c < batch.coord_dim(); c++) {
        coordinates.at({b, sample, c}) =
            batch.coordinates().at({b, source_point, c});
      }

      for (int64_t v = 0; v < batch.value_dim(); v++) {
        values.at({b, sample, v}) = batch.values().at({b, source_point, v});
      }
    }
  }

  return CoordinateBatch(coordinates, values);
}
