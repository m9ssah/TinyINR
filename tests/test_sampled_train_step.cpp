#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

#include "ops/fourier_embedding.h"
#include "training/sampled_train_step.h"

static void assertNear(double actual, double expected, double tolerance = 1e-6) {
  assert(std::isfinite(actual) && std::isfinite(expected));
  assert(std::fabs(actual - expected) <= tolerance * (1.0 + std::fabs(expected)));
}

static void assertSame(const Tensor &a, const Tensor &b) {
  assert(a.shape() == b.shape());
  for (int64_t i = 0; i < a.numel(); ++i) {
    assertNear(a.data()[i], b.data()[i]);
  }
}

static void assertSameParameters(const Mlp &a, const Mlp &b) {
  assert(a.layers.size() == b.layers.size());
  for (size_t i = 0; i < a.layers.size(); ++i) {
    assertSame(a.layers[i].weight, b.layers[i].weight);
    assertSame(a.layers[i].bias, b.layers[i].bias);
  }
}

static TrainStepConfig makeConfig(LossMode mode, bool deterministic = true,
                                  int64_t frequencies = 8) {
  return TrainStepConfig{mode, 2, 3, frequencies, 0.01f, deterministic, 123};
}

static MlpConfig modelConfig(const TrainStepConfig &config) {
  const int64_t input_dim = config.loss_mode == LossMode::MSE
                                ? mseInputDim(2, config.num_frequencies)
                                : cicfmInputDim(2, 3, config.num_frequencies);
  return MlpConfig{input_dim, 4, 3, 3, ActivationKind::SiLU};
}

static CoordinateBatch makeBatch() {
  Tensor coordinates({2, 4, 2});
  Tensor values({2, 4, 3});
  for (int64_t b = 0; b < 2; ++b) {
    for (int64_t n = 0; n < 4; ++n) {
      coordinates.at({b, n, 0}) = -0.8f + 0.4f * n;
      coordinates.at({b, n, 1}) = -0.5f + 0.3f * b;
      for (int64_t c = 0; c < 3; ++c) {
        values.at({b, n, c}) = 0.1f * (b + n + c);
      }
    }
  }
  return CoordinateBatch(coordinates, values);
}

static void testManualParity(LossMode mode, bool deterministic, int64_t frequencies,
                             int64_t count) {
  const auto batch = makeBatch();
  const Tensor original_coordinates = batch.coordinates();
  const Tensor original_values = batch.values();
  const auto config = makeConfig(mode, deterministic, frequencies);
  Mlp composed = createMlp(modelConfig(config), 17);
  Mlp manual = composed;
  std::mt19937 generator(42), manual_rng(42);
  const auto before_rng = generator;
  const auto actual = sampledTrainStep(composed, batch, count, config, generator);

  Tensor priorities({2, 4, 1});
  SamplingConfig sampling;
  sampling.eta = 1.0f;
  const auto selected = sampleCoordinateIndices(priorities, count, sampling, manual_rng);
  const int64_t feature_dim = mseInputDim(2, frequencies);
  Tensor features({2, count, feature_dim});
  Tensor targets({2, count, 3});
  // Assemble each example independently to catch cross-batch flattening errors.
  for (int64_t b = 0; b < 2; ++b) {
    Tensor coordinates({count, 2});
    for (int64_t k = 0; k < count; ++k) {
      const int64_t index = selected.indices[static_cast<size_t>(b * count + k)];
      for (int64_t d = 0; d < 2; ++d) {
        coordinates.at({k, d}) = batch.coordinates().at({b, index, d});
      }
      for (int64_t c = 0; c < 3; ++c) {
        targets.at({b, k, c}) = batch.values().at({b, index, c});
      }
    }
    const Tensor embedded = fourierEmbedding(coordinates, static_cast<int>(frequencies));
    for (int64_t k = 0; k < count; ++k) {
      for (int64_t f = 0; f < feature_dim; ++f) {
        features.at({b, k, f}) = embedded.at({k, f});
      }
    }
  }
  const auto expected = trainStep(manual, features, targets, config);
  assert(actual.selection.indices == selected.indices);
  assert(actual.selection.probabilities == selected.probabilities);
  assert(actual.selection.batch_size == 2 && actual.selection.num_points == 4);
  assert(actual.selection.sample_count == count);
  assert(generator == manual_rng && generator != before_rng);
  assertSame(actual.training.model_input, expected.model_input);
  assertSame(actual.training.model_output, expected.model_output);
  assertSame(actual.training.loss_target, expected.loss_target);
  assertSame(actual.training.output_gradient, expected.output_gradient);
  assertNear(actual.training.loss, expected.loss);
  assertSameParameters(composed, manual);
  for (size_t i = 0; i < composed.layers.size(); ++i) {
    assertSame(composed.layers[i].grad_weight, manual.layers[i].grad_weight);
    assertSame(composed.layers[i].grad_bias, manual.layers[i].grad_bias);
  }
  assert(actual.training.model_input.shape() ==
         std::vector<int64_t>({2, count, modelConfig(config).input_dim}));
  assert(actual.residuals.shape() == std::vector<int64_t>({2, count, 1}));
  double residual_sum = 0.0;
  for (int64_t b = 0; b < 2; ++b) {
    for (int64_t k = 0; k < count; ++k) {
      double error = 0.0;
      for (int64_t c = 0; c < 3; ++c) {
        const double difference = expected.model_output.at({b, k, c}) -
                                  static_cast<double>(expected.loss_target.at({b, k, c}));
        error += difference * difference;
      }
      assertNear(actual.residuals.at({b, k, 0}), error / 3.0);
      residual_sum += actual.residuals.at({b, k, 0});
    }
  }
  assertNear(residual_sum / (2 * count), actual.training.loss);
  assertSame(batch.coordinates(), original_coordinates);
  assertSame(batch.values(), original_values);
}

static void testKnownResiduals(LossMode mode, bool deterministic) {
  Tensor coordinates({2, 4, 2});
  Tensor values({2, 4, 3});
  for (int64_t b = 0; b < 2; ++b) {
    for (int64_t n = 0; n < 4; ++n) {
      for (int64_t c = 0; c < 3; ++c) {
        values.at({b, n, c}) = static_cast<float>(b + c + 1);
      }
    }
  }
  const auto config = makeConfig(mode, deterministic);
  Mlp model(modelConfig(config)); // Zero parameters give known pre-update output.
  std::mt19937 generator(19);
  const auto result = sampledTrainStep(model, CoordinateBatch(coordinates, values),
                                      3, config, generator);
  const Tensor noise = sampleUniform({2, 3, 3}, 0.0f, 1.0f, config.seed);
  for (int64_t b = 0; b < 2; ++b) {
    for (int64_t k = 0; k < 3; ++k) {
      double expected = 0.0;
      for (int64_t c = 0; c < 3; ++c) {
        float target = static_cast<float>(b + c + 1);
        if (mode == LossMode::CICFM && !deterministic) {
          target -= noise.at({b, k, c});
        }
        assertNear(result.training.model_output.at({b, k, c}), 0.0);
        assertNear(result.training.loss_target.at({b, k, c}), target);
        expected += static_cast<double>(target) * target;
      }
      assertNear(result.residuals.at({b, k, 0}), expected / 3.0);
      const double rgb_error = b == 0 ? 14.0 / 3.0 : 29.0 / 3.0;
      if (mode == LossMode::CICFM && !deterministic) {
        assert(result.residuals.at({b, k, 0}) < rgb_error);
      } else {
        assertNear(result.residuals.at({b, k, 0}), rgb_error);
      }
    }
  }
  // Residuals describe the forward pass before SGD, though model parameters changed.
  assert(model.layers.back().bias.data()[0] > 0.0f);
}

static void testInvalidInputs() {
  const auto batch = makeBatch();
  const auto valid = makeConfig(LossMode::MSE);
  const auto initial = createMlp(modelConfig(valid), 17);
  const auto check = [&](const TrainStepConfig &config, int64_t count) {
    Mlp model = initial;
    std::mt19937 generator(42);
    const auto before_rng = generator;
    bool threw = false;
    try {
      (void)sampledTrainStep(model, batch, count, config, generator);
    } catch (const std::invalid_argument &) {
      threw = true;
    }
    assert(threw && generator == before_rng);
    assertSameParameters(model, initial);
  };
  check(valid, 0);
  check(valid, -1);
  check(valid, 5);
  auto config = valid;
  config.coord_dim = 3;
  check(config, 2);
  config = valid;
  config.value_dim = 1;
  check(config, 2);
  for (int64_t frequency : {int64_t(0), int64_t(-1), int64_t(16),
                            std::numeric_limits<int64_t>::max()}) {
    config = valid;
    config.num_frequencies = frequency;
    check(config, 2);
  }
  for (float lr : {0.0f, -1.0f, std::numeric_limits<float>::infinity(),
                   std::numeric_limits<float>::quiet_NaN()}) {
    config = valid;
    config.learning_rate = lr;
    check(config, 2);
  }
  config = valid;
  config.loss_mode = static_cast<LossMode>(-1);
  check(config, 2);
  config.loss_mode = LossMode::CICFM; // Correct config, incompatible MSE model.
  check(config, 2);
}

static Tensor initialPriorities() {
  Tensor priorities({2, 4, 1});
  for (int64_t i = 0; i < priorities.numel(); ++i) {
    priorities.data()[i] = 1.0f;
  }
  return priorities;
}

static void testAdaptiveLoop(LossMode mode, bool deterministic) {
  const auto batch = makeBatch();
  auto config = makeConfig(mode, deterministic);
  Mlp model = createMlp(modelConfig(config), 17);
  Mlp replay = model;
  Tensor priorities = initialPriorities();
  Tensor replay_priorities = priorities;
  std::mt19937 generator(42), replay_rng(42);
  const SamplingConfig sampling;
  bool distribution_changed = false;
  for (int step = 0; step < 5; ++step) {
    config.seed = 123 + 2 * step;
    const Tensor before = priorities;
    const auto before_probs = coordinateSamplingProbabilities(priorities, sampling);
    std::mt19937 expected_rng = generator;
    const auto expected_selection =
        sampleCoordinateIndices(priorities, 3, sampling, expected_rng);
    const auto result = sampledTrainStep(model, batch, 3, config, priorities,
                                         sampling, generator, 0.5f);
    const auto repeated = sampledTrainStep(replay, batch, 3, config, replay_priorities,
                                           sampling, replay_rng, 0.5f);
    assert(result.selection.indices == expected_selection.indices);
    assert(result.selection.probabilities == expected_selection.probabilities);
    assert(result.selection.indices == repeated.selection.indices);
    assert(result.selection.probabilities == repeated.selection.probabilities);
    assert(generator == expected_rng && generator == replay_rng);
    assertNear(result.training.loss, repeated.training.loss);
    assertSame(result.residuals, repeated.residuals);
    assertSameParameters(model, replay);
    assertSame(priorities, replay_priorities);
    for (int64_t b = 0; b < 2; ++b) {
      for (int64_t n = 0; n < 4; ++n) {
        double sum = 0.0;
        int count = 0;
        for (int64_t k = 0; k < 3; ++k) {
          if (result.selection.indices[static_cast<size_t>(b * 3 + k)] == n) {
            sum += result.residuals.at({b, k, 0});
            ++count;
          }
        }
        const double expected = count == 0 ? before.at({b, n, 0})
            : 0.5 * before.at({b, n, 0}) + 0.5 * sum / count;
        assertNear(priorities.at({b, n, 0}), expected);
      }
    }
    const auto after_probs = coordinateSamplingProbabilities(priorities, sampling);
    for (size_t i = 0; i < before_probs.size(); ++i) {
      distribution_changed = distribution_changed ||
                             std::fabs(after_probs[i] - before_probs[i]) > 1e-6;
    }
  }
  assert(distribution_changed);
}

static void testUniformPrioritiesUnchanged() {
  const auto batch = makeBatch();
  const auto config = makeConfig(LossMode::CICFM, false);
  Mlp model = createMlp(modelConfig(config), 17);
  Mlp original_path = model;
  Tensor priorities = initialPriorities();
  priorities.data()[0] = 19.0f;
  const Tensor before = priorities;
  SamplingConfig sampling;
  sampling.eta = 1.0f;
  std::mt19937 generator(42), original_rng(42);
  const auto result = sampledTrainStep(model, batch, 3, config, priorities,
                                       sampling, generator);
  const auto original = sampledTrainStep(original_path, batch, 3, config, original_rng);
  assertSame(priorities, before);
  assert(result.selection.indices == original.selection.indices);
  assert(result.selection.probabilities == original.selection.probabilities);
  assertNear(result.training.loss, original.training.loss);
  assertSame(result.residuals, original.residuals);
  assertSameParameters(model, original_path);
  assert(generator == original_rng);
}

static void testAdaptiveFailures() {
  const auto batch = makeBatch();
  const auto config = makeConfig(LossMode::MSE);
  const auto initial = createMlp(modelConfig(config), 17);
  const auto reject = [&](Tensor priorities, SamplingConfig sampling, float beta) {
    Mlp model = initial;
    const Tensor before = priorities;
    std::mt19937 generator(42);
    const auto original_rng = generator;
    bool threw = false;
    try {
      (void)sampledTrainStep(model, batch, 3, config, priorities, sampling, generator, beta);
    } catch (const std::invalid_argument &) {
      threw = true;
    }
    assert(threw && generator == original_rng);
    assertSameParameters(model, initial);
    assertSame(priorities, before);
  };
  reject(Tensor({1, 4, 1}), SamplingConfig{}, 0.9f);
  reject(initialPriorities(), SamplingConfig{}, 1.0f);
  reject(initialPriorities(), SamplingConfig{}, std::numeric_limits<float>::quiet_NaN());
  SamplingConfig bad;
  bad.eta = 0.0f;
  reject(initialPriorities(), bad, 0.9f);
  Tensor negative = initialPriorities();
  negative.data()[7] = -1.0f;
  reject(negative, SamplingConfig{}, 0.9f);

  for (LossMode mode : {LossMode::MSE, LossMode::CICFM}) {
    const auto overflow_config = makeConfig(mode);
    Mlp model(modelConfig(overflow_config));
    const Mlp before_model = model;
    Tensor values({2, 4, 3});
    for (int64_t i = 0; i < values.numel(); ++i) {
      values.data()[i] = 1e20f;
    }
    const CoordinateBatch overflow_batch(batch.coordinates(), values);
    Tensor priorities = initialPriorities();
    const Tensor before = priorities;
    std::mt19937 generator(42);
    bool threw = false;
    try {
      (void)sampledTrainStep(model, overflow_batch, 3, overflow_config, priorities,
                             SamplingConfig{}, generator);
    } catch (const std::runtime_error &) {
      threw = true;
    }
    assert(threw);
    assertSame(priorities, before);
    assertSameParameters(model, before_model);
  }
}

int main() {
  testManualParity(LossMode::MSE, true, 8, 3);
  testManualParity(LossMode::CICFM, true, 8, 3);
  testManualParity(LossMode::CICFM, false, 8, 3);
  testManualParity(LossMode::MSE, true, 16, 1);
  testManualParity(LossMode::CICFM, false, 16, 4);
  testKnownResiduals(LossMode::MSE, true);
  testKnownResiduals(LossMode::CICFM, true);
  testKnownResiduals(LossMode::CICFM, false);
  testInvalidInputs();
  testAdaptiveLoop(LossMode::MSE, true);
  testAdaptiveLoop(LossMode::CICFM, true);
  testAdaptiveLoop(LossMode::CICFM, false);
  testUniformPrioritiesUnchanged();
  testAdaptiveFailures();
  std::cout << "test_sampled_train_step PASS\n";
}
