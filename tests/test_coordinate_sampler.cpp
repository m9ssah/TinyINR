#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

#include "sampling/coordinate_sampler.h"

static void assertNear(double actual, double expected, double tolerance = 1e-12) {
  assert(std::isfinite(actual));
  assert(std::fabs(actual - expected) <= tolerance);
}

static void testProbabilities() {
  Tensor priorities({2, 3, 1});
  for (int64_t i = 0; i < 3; ++i) {
    priorities.at({1, i, 0}) = 7.0f;
  }
  SamplingConfig config;
  auto probabilities = coordinateSamplingProbabilities(priorities, config);
  for (double p : probabilities) {
    assertNear(p, 1.0 / 3.0); // Both zero and equal positive priorities.
  }

  config.eta = 0.25f;
  config.epsilon = 1.0f;
  for (int64_t i = 0; i < 3; ++i) {
    priorities.at({0, i, 0}) = static_cast<float>(i);
    priorities.at({1, i, 0}) = static_cast<float>(2 - i);
  }
  probabilities = coordinateSamplingProbabilities(priorities, config);
  const double expected[] = {5.0 / 24.0, 8.0 / 24.0, 11.0 / 24.0};
  for (size_t i = 0; i < 3; ++i) {
    assertNear(probabilities[i], expected[i]);
    assertNear(probabilities[3 + i], expected[2 - i]);
  }
  for (size_t b = 0; b < 2; ++b) {
    assertNear(std::accumulate(probabilities.begin() + b * 3,
                               probabilities.begin() + (b + 1) * 3, 0.0), 1.0);
    for (size_t i = 0; i < 3; ++i) {
      assert(probabilities[b * 3 + i] >= config.eta / 3.0 - 1e-12);
    }
  }

  priorities.at({1, 0, 0}) = 100.0f;
  auto changed = coordinateSamplingProbabilities(priorities, config);
  for (size_t i = 0; i < 3; ++i) {
    assertNear(changed[i], probabilities[i]); // Other batch row is unaffected.
  }
  assert(changed[3] > probabilities[3]);

  config.eta = 1.0f;
  for (double p : coordinateSamplingProbabilities(priorities, config)) {
    assertNear(p, 1.0 / 3.0);
  }
}

static void testExtremePriorities() {
  Tensor priorities({1, 3, 1});
  priorities.data()[0] = std::numeric_limits<float>::max();
  priorities.data()[1] = std::numeric_limits<float>::denorm_min();
  SamplingConfig config;
  auto probabilities = coordinateSamplingProbabilities(priorities, config);
  assertNear(std::accumulate(probabilities.begin(), probabilities.end(), 0.0), 1.0);
  for (double p : probabilities) {
    assert(p >= config.eta / 3.0 - 1e-12);
  }
  config.epsilon = std::numeric_limits<float>::max();
  probabilities = coordinateSamplingProbabilities(priorities, config);
  assertNear(std::accumulate(probabilities.begin(), probabilities.end(), 0.0), 1.0);

  config.eta = std::numeric_limits<float>::denorm_min();
  config.epsilon = std::numeric_limits<float>::denorm_min();
  for (double p : coordinateSamplingProbabilities(priorities, config)) {
    assert(std::isfinite(p) && p > 0.0);
  }
}

static void testDraws() {
  Tensor priorities({2, 3, 1});
  for (int64_t i = 0; i < 3; ++i) {
    priorities.at({0, i, 0}) = static_cast<float>(i);
    priorities.at({1, i, 0}) = static_cast<float>(2 - i);
  }
  SamplingConfig config;
  config.eta = 0.25f;
  config.epsilon = 1.0f;
  const auto expected = coordinateSamplingProbabilities(priorities, config);
  std::mt19937 generator(42), replay(42);
  const std::mt19937 original = generator;
  int counts[2][3] = {};
  bool saw_duplicate = false;
  const int trials = 10000;
  for (int trial = 0; trial < trials; ++trial) {
    const auto selection = sampleCoordinateIndices(priorities, 3, config, generator);
    const auto repeated = sampleCoordinateIndices(priorities, 3, config, replay);
    assert(selection.batch_size == 2 && selection.num_points == 3);
    assert(selection.sample_count == 3 && selection.indices.size() == 6);
    assert(selection.probabilities.size() == 6);
    assert(selection.indices == repeated.indices);
    assert(selection.probabilities == repeated.probabilities);
    for (size_t b = 0; b < 2; ++b) {
      bool seen[3] = {};
      for (size_t k = 0; k < 3; ++k) {
        const size_t offset = b * 3 + k;
        const int64_t index = selection.indices[offset];
        assert(index >= 0 && index < 3);
        assertNear(selection.probabilities[offset], expected[b * 3 + index]);
        saw_duplicate = saw_duplicate || seen[index];
        seen[index] = true;
        ++counts[b][index];
      }
    }
  }
  assert(generator != original && generator == replay);
  assert(saw_duplicate);
  for (size_t b = 0; b < 2; ++b) {
    for (size_t i = 0; i < 3; ++i) {
      const double frequency = counts[b][i] / static_cast<double>(trials * 3);
      assertNear(frequency, expected[b * 3 + i], 0.02);
    }
  }
  assert(counts[0][2] > counts[0][0]);
  assert(counts[1][0] > counts[1][2]);

  config.eta = 1.0f;
  int uniform_counts[3] = {};
  for (int trial = 0; trial < trials; ++trial) {
    const auto selection = sampleCoordinateIndices(priorities, 1, config, generator);
    assert(selection.indices.size() == 2 && selection.sample_count == 1);
    for (double p : selection.probabilities) {
      assertNear(p, 1.0 / 3.0);
    }
    ++uniform_counts[selection.indices[0]];
  }
  for (int count : uniform_counts) {
    assertNear(count / static_cast<double>(trials), 1.0 / 3.0, 0.03);
  }
  assert(counts[0][2] / static_cast<double>(trials * 3) >
         uniform_counts[2] / static_cast<double>(trials));

  Tensor single({2, 1, 1});
  const auto selection = sampleCoordinateIndices(single, 1, config, generator);
  assert(selection.indices == std::vector<int64_t>({0, 0}));
  assert(selection.probabilities == std::vector<double>({1.0, 1.0}));
  for (int64_t i = 0; i < 3; ++i) {
    assert(priorities.at({0, i, 0}) == static_cast<float>(i));
    assert(priorities.at({1, i, 0}) == static_cast<float>(2 - i));
  }
}

static void expectInvalid(const Tensor &priorities, int64_t count,
                          const SamplingConfig &config) {
  std::mt19937 generator(17);
  const auto before = generator;
  bool threw = false;
  try {
    (void)sampleCoordinateIndices(priorities, count, config, generator);
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  assert(threw && generator == before);
}

static void testInvalidInputs() {
  Tensor priorities({2, 3, 1});
  SamplingConfig config;
  expectInvalid(Tensor({3}), 1, config);
  expectInvalid(Tensor({2, 3}), 1, config);
  expectInvalid(Tensor({2, 3, 2}), 1, config);
  expectInvalid(priorities, 0, config);
  expectInvalid(priorities, -1, config);
  expectInvalid(priorities, 4, config);
  expectInvalid(priorities, std::numeric_limits<int64_t>::max(), config);

  const float nan = std::numeric_limits<float>::quiet_NaN();
  const float inf = std::numeric_limits<float>::infinity();
  for (float eta : {0.0f, -1.0f, 1.01f, nan, inf, -inf}) {
    config.eta = eta;
    expectInvalid(priorities, 2, config);
  }
  config = SamplingConfig{};
  for (float epsilon : {0.0f, -1.0f, nan, inf, -inf}) {
    config.epsilon = epsilon;
    expectInvalid(priorities, 2, config);
  }
  config = SamplingConfig{};
  for (float priority : {-1.0f, nan, inf, -inf}) {
    priorities.at({1, 2, 0}) = priority; // Validate later rows before any draw.
    expectInvalid(priorities, 2, config);
    config.eta = 1.0f;
    expectInvalid(priorities, 2, config); // Uniform mode validates too.
    config = SamplingConfig{};
  }
}

static void testPriorityUpdates() {
  Tensor priorities({2, 4, 1});
  for (int64_t i = 0; i < priorities.numel(); ++i) {
    priorities.data()[i] = 2.0f;
  }
  const Tensor initial = priorities;
  SampleSelection selection{2, 4, 3, {1, 1, 3, 0, 2, 2},
                             std::vector<double>(6, 0.25)};
  Tensor residuals({2, 3, 1});
  const float errors[] = {2, 6, 10, 0, 4, 8};
  for (size_t i = 0; i < 6; ++i) {
    residuals.data()[i] = errors[i];
  }
  updateCoordinatePriorities(priorities, selection, residuals, 0.5f);
  const float expected[] = {2, 3, 2, 6, 1, 2, 4, 2};
  for (size_t i = 0; i < 8; ++i) {
    assertNear(priorities.data()[i], expected[i]);
  }
  const auto probabilities = coordinateSamplingProbabilities(priorities, SamplingConfig{});
  assert(probabilities[3] > 0.25);
  assert(probabilities[6] > 0.25);

  Tensor reordered = initial;
  std::swap(selection.indices[0], selection.indices[2]);
  std::swap(residuals.data()[0], residuals.data()[2]);
  std::swap(selection.indices[3], selection.indices[5]);
  std::swap(residuals.data()[3], residuals.data()[5]);
  updateCoordinatePriorities(reordered, selection, residuals, 0.5f);
  for (size_t i = 0; i < 8; ++i) {
    assertNear(reordered.data()[i], expected[i]);
  }

  Tensor replacement = initial;
  updateCoordinatePriorities(replacement, selection, residuals, 0.0f);
  assertNear(replacement.at({0, 1, 0}), 4.0);
  assertNear(replacement.at({1, 2, 0}), 6.0);
  assertNear(replacement.at({1, 0, 0}), 0.0);
  assertNear(replacement.at({0, 0, 0}), 2.0);

  Tensor single({1, 1, 1});
  single.data()[0] = 2.0f;
  Tensor error({1, 1, 1});
  error.data()[0] = 12.0f;
  const SampleSelection one{1, 1, 1, {0}, {1.0}};
  updateCoordinatePriorities(single, one, error);
  assertNear(single.data()[0], 3.0, 1e-6); // Default beta is 0.9.
  single.data()[0] = error.data()[0] = std::numeric_limits<float>::max();
  updateCoordinatePriorities(single, one, error);
  assert(single.data()[0] == std::numeric_limits<float>::max());
}

static void testInvalidUpdatesAreAtomic() {
  Tensor priorities({2, 4, 1});
  for (int64_t i = 0; i < priorities.numel(); ++i) {
    priorities.data()[i] = 2.0f;
  }
  const SampleSelection valid{2, 4, 3, {1, 1, 3, 0, 2, 2},
                              std::vector<double>(6, 0.25)};
  Tensor residuals({2, 3, 1});
  const auto reject = [](Tensor p, const SampleSelection &s, const Tensor &r, float beta) {
    const Tensor before = p;
    bool threw = false;
    try {
      updateCoordinatePriorities(p, s, r, beta);
    } catch (const std::invalid_argument &) {
      threw = true;
    }
    assert(threw);
    assert(p.shape() == before.shape());
    assert(std::memcmp(p.data(), before.data(),
                       static_cast<size_t>(p.numel()) * sizeof(float)) == 0);
  };
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const float inf = std::numeric_limits<float>::infinity();
  for (float beta : {-0.1f, 1.0f, inf, nan}) {
    reject(priorities, valid, residuals, beta);
  }
  for (float bad : {-1.0f, inf, nan}) {
    Tensor bad_residuals = residuals;
    bad_residuals.data()[5] = bad; // An invalid last item must not update earlier ones.
    reject(priorities, valid, bad_residuals, 0.5f);
    Tensor bad_priorities = priorities;
    bad_priorities.data()[7] = bad; // Unobserved priorities must be validated too.
    reject(bad_priorities, valid, residuals, 0.5f);
  }
  reject(Tensor({2, 4}), valid, residuals, 0.5f);
  reject(priorities, valid, Tensor({2, 3}), 0.5f);
  reject(priorities, valid, Tensor({2, 3, 2}), 0.5f);
  auto selection = valid;
  selection.batch_size = 1;
  reject(priorities, selection, residuals, 0.5f);
  selection = valid;
  selection.num_points = 5;
  reject(priorities, selection, residuals, 0.5f);
  for (int64_t k : {int64_t(0), int64_t(-1), int64_t(5)}) {
    selection = valid;
    selection.sample_count = k;
    reject(priorities, selection, residuals, 0.5f);
  }
  for (int64_t index : {int64_t(-1), int64_t(4)}) {
    selection = valid;
    selection.indices.back() = index;
    reject(priorities, selection, residuals, 0.5f);
  }
  selection = valid;
  selection.indices.pop_back();
  reject(priorities, selection, residuals, 0.5f);
  selection = valid;
  selection.probabilities.push_back(0.25);
  reject(priorities, selection, residuals, 0.5f);
  for (double probability : {0.0, -1.0, 1.1,
                             std::numeric_limits<double>::quiet_NaN(),
                             std::numeric_limits<double>::infinity()}) {
    selection = valid;
    selection.probabilities.back() = probability;
    reject(priorities, selection, residuals, 0.5f);
  }
}

int main() {
  testProbabilities();
  testExtremePriorities();
  testDraws();
  testInvalidInputs();
  testPriorityUpdates();
  testInvalidUpdatesAreAtomic();
  std::cout << "test_coordinate_sampler PASS\n";
}
