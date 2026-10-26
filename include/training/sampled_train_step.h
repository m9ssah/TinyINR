#pragma once

#include "ops/coordinate_batch.h"
#include "sampling/coordinate_sampler.h"
#include "training/train_step.h"

struct SampledTrainStepResult {
  SampleSelection selection;
  TrainStepResult training;
  Tensor residuals; // [B,K,1]: channel-mean squared error before the SGD update.
};

// Wall-clock milliseconds inside the step, excluding caller evaluation/logging.
struct SampledTrainTimings {
  double sampling_ms = 0;
  double gather_ms = 0;
  double embedding_ms = 0;
  double train_ms = 0;
  double residual_ms = 0;
  double update_ms = 0;
  double step_ms = 0;
};

// Uniform draws with replacement, followed by gather, CPU embedding, and one
// existing trainStep. The caller owns the coordinate RNG and config.seed controls
// CICFM noise/time separately. No priority updates or importance correction.
// Requires 1 <= sample_count <= N and matching batch/config/model dimensions.
// Rejects invalid configuration before drawing; later failures do not roll back
// the RNG or the model. The source batch is never modified.
SampledTrainStepResult sampledTrainStep(Mlp &model, const CoordinateBatch &batch,
                                      int64_t sample_count,
                                      const TrainStepConfig &config,
                                      std::mt19937 &generator);

// Uses caller-owned [B,N,1] priorities (initialize to ones) to select points.
// After successful training, updates observed priorities from returned residuals.
// eta=1 gives uniform draws and leaves priorities unchanged. Probabilities in the
// result describe the draw before the update. Loss is not importance-corrected.
// Priority updates are atomic, but later failures do not roll back model/RNG state.
SampledTrainStepResult sampledTrainStep(Mlp &model, const CoordinateBatch &batch,
                                      int64_t sample_count,
                                      const TrainStepConfig &config,
                                      Tensor &priorities,
                                      const SamplingConfig &sampling,
                                      std::mt19937 &generator,
                                      float beta = 0.90f,
                                      SampledTrainTimings *timings = nullptr);
