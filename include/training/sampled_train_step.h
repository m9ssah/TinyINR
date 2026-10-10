#pragma once

#include "ops/coordinate_batch.h"
#include "sampling/coordinate_sampler.h"
#include "training/train_step.h"

struct SampledTrainStepResult {
  SampleSelection selection;
  TrainStepResult training;
  Tensor residuals; // [B,K,1]: channel-mean squared error before the SGD update.
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
