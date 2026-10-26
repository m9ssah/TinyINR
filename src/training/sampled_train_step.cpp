#include "training/sampled_train_step.h"

#include <cmath>
#include <chrono>
#include <limits>
#include <stdexcept>
#include <utility>

#include "ops/fourier_embedding.h"

SampledTrainStepResult sampledTrainStep(Mlp &model, const CoordinateBatch &batch,
                                      int64_t sample_count,
                                      const TrainStepConfig &config,
                                      Tensor &priorities,
                                      const SamplingConfig &sampling,
                                      std::mt19937 &generator, float beta,
                                      SampledTrainTimings *timings) {
  using Clock = std::chrono::steady_clock;
  const auto start = Clock::now();
  if (timings) {
    *timings = SampledTrainTimings{};
  }
  if (sample_count <= 0 || sample_count > batch.num_points() ||
      config.coord_dim != batch.coord_dim() ||
      config.value_dim != batch.value_dim() ||
      !std::isfinite(config.learning_rate) || config.learning_rate <= 0.0f) {
    throw std::invalid_argument("invalid sampled train-step configuration");
  }
  if (config.loss_mode != LossMode::MSE && config.loss_mode != LossMode::CICFM) {
    throw std::invalid_argument("unknown loss mode");
  }

  // The existing raw CPU embedding uses int for dimensions and flat offsets.
  const int64_t max_int = std::numeric_limits<int>::max();
  if (config.num_frequencies <= 0 ||
      config.num_frequencies >= std::numeric_limits<float>::max_exponent ||
      config.coord_dim > max_int ||
      config.num_frequencies > (max_int / config.coord_dim - 1) / 2) {
    throw std::invalid_argument("feature width exceeds CPU embedding limits");
  }
  const int64_t feature_dim = mseInputDim(config.coord_dim, config.num_frequencies);
  if (sample_count > max_int / feature_dim ||
      batch.batch_size() > max_int / feature_dim / sample_count) {
    throw std::invalid_argument("sampled batch exceeds CPU embedding limits");
  }
  if (config.loss_mode == LossMode::CICFM &&
      config.value_dim > std::numeric_limits<int64_t>::max() - feature_dim - 1) {
    throw std::invalid_argument("CICFM input width overflow");
  }
  const int64_t input_dim = config.loss_mode == LossMode::MSE
                                ? feature_dim
                                : feature_dim + config.value_dim + 1;
  if (model.config.input_dim != input_dim ||
      model.config.output_dim != config.value_dim) {
    throw std::invalid_argument("sampled train-step model dimensions mismatch");
  }

  if (priorities.shape() != std::vector<int64_t>(
          {batch.batch_size(), batch.num_points(), 1}) ||
      !std::isfinite(beta) || beta < 0.0f || beta >= 1.0f) {
    throw std::invalid_argument("invalid adaptive training priorities or beta");
  }
  const auto sampling_start = Clock::now();
  SampleSelection selection =
      sampleCoordinateIndices(priorities, sample_count, sampling, generator);
  const auto gather_start = Clock::now();
  CoordinateBatch sampled = gatherCoordinateBatch(batch, selection.indices, sample_count);
  const auto embedding_start = Clock::now();
  const int64_t rows = batch.batch_size() * sample_count;
  Tensor embedded = fourierEmbedding(
      sampled.coordinates().reshape({rows, config.coord_dim}),
      static_cast<int>(config.num_frequencies));
  Tensor features = embedded.reshape({batch.batch_size(), sample_count, feature_dim});
  const auto train_start = Clock::now();
  TrainStepResult training = trainStep(model, features, sampled.values(), config);
  const auto residual_start = Clock::now();

  Tensor residuals({batch.batch_size(), sample_count, 1});
  for (int64_t b = 0; b < batch.batch_size(); ++b) {
    for (int64_t k = 0; k < sample_count; ++k) {
      double sum = 0.0;
      for (int64_t c = 0; c < config.value_dim; ++c) {
        const double difference =
            static_cast<double>(training.model_output.at({b, k, c})) -
            training.loss_target.at({b, k, c});
        sum += difference * difference;
      }
      residuals.at({b, k, 0}) = static_cast<float>(sum / config.value_dim);
    }
  }
  const auto update_start = Clock::now();
  if (sampling.eta < 1.0f) {
    updateCoordinatePriorities(priorities, selection, residuals, beta);
  }
  const auto end = Clock::now();
  if (timings) {
    const auto ms = [](Clock::time_point from, Clock::time_point to) {
      return std::chrono::duration<double, std::milli>(to - from).count();
    };
    timings->sampling_ms = ms(sampling_start, gather_start);
    timings->gather_ms = ms(gather_start, embedding_start);
    timings->embedding_ms = ms(embedding_start, train_start);
    timings->train_ms = ms(train_start, residual_start);
    timings->residual_ms = ms(residual_start, update_start);
    timings->update_ms = sampling.eta < 1.0f ? ms(update_start, end) : 0.0;
    timings->step_ms = ms(start, end);
  }
  return SampledTrainStepResult{std::move(selection), std::move(training),
                                std::move(residuals)};
}

SampledTrainStepResult sampledTrainStep(Mlp &model, const CoordinateBatch &batch,
                                      int64_t sample_count,
                                      const TrainStepConfig &config,
                                      std::mt19937 &generator) {
  Tensor priorities({batch.batch_size(), batch.num_points(), 1});
  SamplingConfig sampling;
  sampling.eta = 1.0f;
  return sampledTrainStep(model, batch, sample_count, config, priorities, sampling,
                          generator);
}
