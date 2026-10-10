#include "sampling/coordinate_sampler.h"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace {

size_t checkedCount(int64_t rows, int64_t columns) {
  if (rows <= 0 || columns <= 0) {
    throw std::invalid_argument("sampling dimensions must be positive");
  }
  const uint64_t max_size = std::numeric_limits<size_t>::max();
  if (static_cast<uint64_t>(rows) >
      max_size / static_cast<uint64_t>(columns)) {
    throw std::length_error("sampling size overflow");
  }
  const size_t count = static_cast<size_t>(rows) * static_cast<size_t>(columns);
  if (count > std::vector<double>().max_size() ||
      count > std::vector<int64_t>().max_size()) {
    throw std::length_error("sampling size exceeds vector capacity");
  }
  return count;
}

void validateShape(const Tensor &priorities) {
  if (priorities.ndim() != 3 || priorities.shape()[2] != 1) {
    throw std::invalid_argument("priorities must have shape [B,N,1]");
  }
}

} // namespace

std::vector<double> coordinateSamplingProbabilities(
    const Tensor &priorities, const SamplingConfig &config) {
  validateShape(priorities);
  if (!std::isfinite(config.eta) || config.eta <= 0.0f || config.eta > 1.0f ||
      !std::isfinite(config.epsilon) || config.epsilon <= 0.0f) {
    throw std::invalid_argument("invalid sampling configuration");
  }

  const int64_t batch_size = priorities.shape()[0];
  const int64_t num_points = priorities.shape()[1];
  const size_t count = checkedCount(batch_size, num_points);
  for (size_t i = 0; i < count; ++i) {
    if (!std::isfinite(priorities.data()[i]) || priorities.data()[i] < 0.0f) {
      throw std::invalid_argument("priorities must be finite and nonnegative");
    }
  }

  std::vector<double> probabilities(count);
  const size_t width = static_cast<size_t>(num_points);
  const double eta = config.eta;
  const double epsilon = config.epsilon;
  for (int64_t b = 0; b < batch_size; ++b) {
    const size_t offset = static_cast<size_t>(b) * width;
    double total = 0.0;
    for (size_t i = 0; i < width; ++i) {
      total += static_cast<double>(priorities.data()[offset + i]) + epsilon;
    }
    for (size_t i = 0; i < width; ++i) {
      const double priority = priorities.data()[offset + i];
      probabilities[offset + i] =
          eta / static_cast<double>(num_points) +
          (1.0 - eta) * ((priority + epsilon) / total);
    }
  }
  return probabilities;
}

SampleSelection sampleCoordinateIndices(const Tensor &priorities,
                                        int64_t sample_count,
                                        const SamplingConfig &config,
                                        std::mt19937 &generator) {
  validateShape(priorities);
  const int64_t batch_size = priorities.shape()[0];
  const int64_t num_points = priorities.shape()[1];
  if (sample_count <= 0 || sample_count > num_points) {
    throw std::invalid_argument("sample_count must be in [1,N]");
  }
  const size_t count = checkedCount(batch_size, sample_count);
  const std::vector<double> probabilities =
      coordinateSamplingProbabilities(priorities, config);
  SampleSelection selection{batch_size, num_points, sample_count,
                            std::vector<int64_t>(count),
                            std::vector<double>(count)};
  const size_t width = static_cast<size_t>(num_points);
  const size_t draws = static_cast<size_t>(sample_count);
  for (int64_t b = 0; b < batch_size; ++b) {
    const size_t offset = static_cast<size_t>(b) * width;
    std::discrete_distribution<int64_t> distribution(
        probabilities.data() + offset, probabilities.data() + offset + width);
    const std::vector<double> normalized = distribution.probabilities();
    for (size_t k = 0; k < draws; ++k) {
      const size_t destination = static_cast<size_t>(b) * draws + k;
      const int64_t index = distribution(generator);
      selection.indices[destination] = index;
      selection.probabilities[destination] = normalized[static_cast<size_t>(index)];
    }
  }
  return selection;
}
