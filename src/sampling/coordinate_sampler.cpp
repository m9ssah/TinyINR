#include "sampling/coordinate_sampler.h"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

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

size_t validatePriorities(const Tensor &priorities) {
  validateShape(priorities);
  const size_t count = checkedCount(priorities.shape()[0], priorities.shape()[1]);
  for (size_t i = 0; i < count; ++i) {
    if (!std::isfinite(priorities.data()[i]) || priorities.data()[i] < 0.0f) {
      throw std::invalid_argument("priorities must be finite and nonnegative");
    }
  }
  return count;
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
  const size_t count = validatePriorities(priorities);

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

void updateCoordinatePriorities(Tensor &priorities,
                                const SampleSelection &selection,
                                const Tensor &residuals, float beta) {
  const size_t count = validatePriorities(priorities);
  if (!std::isfinite(beta) || beta < 0.0f || beta >= 1.0f ||
      selection.batch_size != priorities.shape()[0] ||
      selection.num_points != priorities.shape()[1] ||
      selection.sample_count <= 0 || selection.sample_count > selection.num_points) {
    throw std::invalid_argument("invalid priority update configuration");
  }
  const size_t draws = checkedCount(selection.batch_size, selection.sample_count);
  if (selection.indices.size() != draws || selection.probabilities.size() != draws ||
      residuals.shape() != std::vector<int64_t>(
          {selection.batch_size, selection.sample_count, 1})) {
    throw std::invalid_argument("priority update shapes do not match selection");
  }
  for (size_t i = 0; i < draws; ++i) {
    if (selection.indices[i] < 0 || selection.indices[i] >= selection.num_points ||
        !std::isfinite(selection.probabilities[i]) ||
        selection.probabilities[i] <= 0.0 || selection.probabilities[i] > 1.0 ||
        !std::isfinite(residuals.data()[i]) || residuals.data()[i] < 0.0f) {
      throw std::invalid_argument("invalid priority update observation");
    }
  }

  std::vector<double> sums(count, 0.0);
  std::vector<int64_t> observations(count, 0);
  const size_t width = static_cast<size_t>(selection.num_points);
  const size_t samples = static_cast<size_t>(selection.sample_count);
  for (size_t i = 0; i < draws; ++i) {
    const size_t original = (i / samples) * width +
                            static_cast<size_t>(selection.indices[i]);
    sums[original] += residuals.data()[i];
    ++observations[original];
  }

  Tensor updated = priorities;
  for (size_t i = 0; i < count; ++i) {
    if (observations[i] == 0) {
      continue;
    }
    const double mean = sums[i] / static_cast<double>(observations[i]);
    const double value = static_cast<double>(beta) * priorities.data()[i] +
                         (1.0 - static_cast<double>(beta)) * mean;
    if (!std::isfinite(value) || value > std::numeric_limits<float>::max()) {
      throw std::runtime_error("non-finite updated priority");
    }
    updated.data()[i] = static_cast<float>(value);
    if (!std::isfinite(updated.data()[i])) {
      throw std::runtime_error("non-finite stored priority");
    }
  }
  priorities = std::move(updated);
}
