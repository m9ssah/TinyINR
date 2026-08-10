#pragma once

#include <cstdint>

#include "../model/mlp.h"
#include "../tensor.h"
#include "cicfm.h"
#include "loss.h"
#include "train_debug.h"

struct TrainStepConfig {
  LossMode loss_mode;
  int64_t coord_dim;
  int64_t value_dim;
  int64_t num_frequencies;
  float learning_rate;
  bool deterministic_cicfm;
  uint32_t seed;
};

struct TrainStepResult {
  float loss;
  Tensor model_input;
  Tensor model_output;
  Tensor loss_target;
  Tensor output_gradient;
  TensorStats output_stats;
  TensorStats gradient_stats;
};

TrainStepResult trainStep(Mlp &model, const Tensor &features,
                          const Tensor &targets,
                          const TrainStepConfig &config);
