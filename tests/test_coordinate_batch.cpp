#include <cassert>
#include <iostream>
#include <limits>
#include <set>
#include <stdexcept>

#include "ops/coordinate_batch.h"
#include "sampling/coordinate_sampler.h"

static CoordinateBatch makeMarkerBatch(int64_t batches = 2, int64_t points = 4,
                                       int64_t coord_dim = 2,
                                       int64_t value_dim = 3) {
  Tensor coordinates({batches, points, coord_dim});
  Tensor values({batches, points, value_dim});
  for (int64_t b = 0; b < batches; ++b) {
    for (int64_t n = 0; n < points; ++n) {
      for (int64_t c = 0; c < coord_dim; ++c) {
        coordinates.at({b, n, c}) = static_cast<float>(1000 * b + 10 * n + c);
      }
      for (int64_t v = 0; v < value_dim; ++v) {
        values.at({b, n, v}) = static_cast<float>(5000 + 1000 * b + 10 * n + v);
      }
    }
  }
  return CoordinateBatch(coordinates, values);
}

static void assertSame(const Tensor &actual, const Tensor &expected) {
  assert(actual.shape() == expected.shape());
  for (int64_t i = 0; i < actual.numel(); ++i) {
    assert(actual.data()[i] == expected.data()[i]);
  }
}

static void checkGather(const CoordinateBatch &source,
                        const std::vector<int64_t> &indices, int64_t count) {
  const Tensor before_coordinates = source.coordinates();
  const Tensor before_values = source.values();
  const auto before_indices = indices;
  const auto gathered = gatherCoordinateBatch(source, indices, count);
  assert(gathered.coordinates().shape() ==
         std::vector<int64_t>({source.batch_size(), count, source.coord_dim()}));
  assert(gathered.values().shape() ==
         std::vector<int64_t>({source.batch_size(), count, source.value_dim()}));
  for (int64_t b = 0; b < source.batch_size(); ++b) {
    for (int64_t k = 0; k < count; ++k) {
      const int64_t original = indices[static_cast<size_t>(b * count + k)];
      for (int64_t c = 0; c < source.coord_dim(); ++c) {
        assert(gathered.coordinates().at({b, k, c}) ==
               static_cast<float>(1000 * b + 10 * original + c));
      }
      for (int64_t v = 0; v < source.value_dim(); ++v) {
        assert(gathered.values().at({b, k, v}) ==
               static_cast<float>(5000 + 1000 * b + 10 * original + v));
      }
    }
  }
  assertSame(source.coordinates(), before_coordinates);
  assertSame(source.values(), before_values);
  assert(indices == before_indices);
  assert(gathered.coordinates().data() != source.coordinates().data());
  assert(gathered.values().data() != source.values().data());
}

static void testGather() {
  const auto batch = makeMarkerBatch();
  checkGather(batch, {3, 1, 3, 0, 2, 0}, 3); // Reordering and duplicates per row.
  checkGather(batch, {2, 0}, 1);
  checkGather(batch, {3, 2, 1, 0, 1, 3, 0, 2}, 4);
  checkGather(batch, {0, 1, 2, 3, 0, 1, 2, 3}, 4);
  checkGather(makeMarkerBatch(1, 1, 3, 1), {0}, 1);
  checkGather(makeMarkerBatch(2, 4, 3, 1), {1, 1, 2, 0}, 2);

  const auto image = generateCoordinateBatch(
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, 2, 2, 3);
  const auto gathered = gatherCoordinateBatch(image, {3, 0, 3}, 3);
  assert(gathered.coordinates().at({0, 0, 0}) == 1.0f);
  assert(gathered.coordinates().at({0, 0, 1}) == 1.0f);
  assert(gathered.coordinates().at({0, 1, 0}) == -1.0f);
  assert(gathered.coordinates().at({0, 1, 1}) == -1.0f);
  for (int64_t v = 0; v < 3; ++v) {
    assert(gathered.values().at({0, 0, v}) == 10.0f + v);
    assert(gathered.values().at({0, 1, v}) == 1.0f + v);
    assert(gathered.values().at({0, 2, v}) == 10.0f + v);
  }
}

static void testSamplerIntegration() {
  const auto batch = makeMarkerBatch();
  Tensor priorities({2, 4, 1});
  priorities.at({0, 3, 0}) = 10.0f;
  priorities.at({1, 0, 0}) = 10.0f;
  std::mt19937 generator(42);
  SamplingConfig config;
  for (float eta : {0.1f, 1.0f}) {
    config.eta = eta;
    const auto selection = sampleCoordinateIndices(priorities, 3, config, generator);
    const auto probabilities = selection.probabilities;
    checkGather(batch, selection.indices, selection.sample_count);
    assert(selection.probabilities == probabilities);
  }
}

static void testInvalidInputs() {
  const auto batch = makeMarkerBatch();
  const Tensor before_coordinates = batch.coordinates();
  const Tensor before_values = batch.values();
  const auto expectInvalid = [&](const std::vector<int64_t> &indices,
                                 int64_t count) {
    const auto before_indices = indices;
    bool threw = false;
    try {
      (void)gatherCoordinateBatch(batch, indices, count);
    } catch (const std::invalid_argument &) {
      threw = true;
    }
    assert(threw);
    assertSame(batch.coordinates(), before_coordinates);
    assertSame(batch.values(), before_values);
    assert(indices == before_indices);
  };
  expectInvalid({}, 0);
  expectInvalid({}, -1);
  expectInvalid({}, 5);
  expectInvalid({}, std::numeric_limits<int64_t>::max());
  expectInvalid({}, 1);
  expectInvalid({0, 1, 2}, 2); // Too short for B=2, K=2.
  expectInvalid({0, 1, 2, 3, 0}, 2); // Too long.
  expectInvalid({-1, 0, 1, 2}, 2);
  expectInvalid({0, 1, 2, 4}, 2); // Invalid index in the last batch row.
  expectInvalid({0, std::numeric_limits<int64_t>::max()}, 1);
}

static void testExistingRandomSubsample() {
  const auto batch = makeMarkerBatch();
  const auto sampled = randomSubsample(batch, 4);
  assert(sampled.num_points() == 4 && sampled.batch_size() == 2);
  for (int64_t b = 0; b < 2; ++b) {
    std::set<int64_t> seen;
    for (int64_t k = 0; k < 4; ++k) {
      const int64_t index =
          static_cast<int64_t>(sampled.coordinates().at({b, k, 0}) - 1000 * b) / 10;
      assert(index >= 0 && index < 4);
      assert(seen.insert(index).second); // Original path stays without replacement.
      for (int64_t v = 0; v < 3; ++v) {
        assert(sampled.values().at({b, k, v}) == batch.values().at({b, index, v}));
      }
    }
  }
}

int main() {
  testGather();
  testSamplerIntegration();
  testInvalidInputs();
  testExistingRandomSubsample();
  std::cout << "test_coordinate_batch PASS\n";
}
