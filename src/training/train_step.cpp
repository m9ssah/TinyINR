#include "training/train_step.h"

#include <stdexcept>

namespace {

void require3d(const Tensor &tensor, const char *name) {
  if (tensor.ndim() != 3) {
    throw std::invalid_argument(std::string(name) + " must be 3D");
  }
}

void validateSharedInputs(const Tensor &features, const Tensor &targets,
                          const TrainStepConfig &config) {
  require3d(features, "features");
  require3d(targets, "targets");

  if (features.shape()[0] != targets.shape()[0] ||
      features.shape()[1] != targets.shape()[1]) {
    throw std::invalid_argument(
        "features and targets must share batch and point dimensions");
  }
  if (targets.shape()[2] != config.value_dim) {
    throw std::invalid_argument("targets value dimension mismatch");
  }
  if (config.learning_rate <= 0.0f) {
    throw std::invalid_argument("learning_rate must be positive");
  }
}

Tensor makeRandomTime(const Tensor &targets, uint32_t seed) {
  return sampleUniform({targets.shape()[0], targets.shape()[1], 1}, 0.0f, 1.0f,
                       seed);
}

CicfmBatch makeConfiguredCicfmBatch(const Tensor &features,
                                    const Tensor &targets,
                                    const TrainStepConfig &config) {
  if (config.deterministic_cicfm) {
    return makeDeterministicCicfmBatch(features, targets);
  }

  Tensor z0 = sampleUniform(targets.shape(), 0.0f, 1.0f, config.seed);
  Tensor t = makeRandomTime(targets, config.seed + 1);
  return makeCicfmBatch(features, targets, z0, t);
}

} // namespace

TrainStepResult trainStep(Mlp &model, const Tensor &features,
                          const Tensor &targets,
                          const TrainStepConfig &config) {
  validateSharedInputs(features, targets, config);

  Tensor model_input = features;
  Tensor loss_target = targets;

  if (config.loss_mode == LossMode::MSE) {
    const int64_t expected_dim =
        mseInputDim(config.coord_dim, config.num_frequencies);
    if (features.shape()[2] != expected_dim ||
        model.config.input_dim != expected_dim) {
      throw std::invalid_argument("MSE input dimension mismatch");
    }
  } else if (config.loss_mode == LossMode::CICFM) {
    const int64_t expected_feature_dim =
        mseInputDim(config.coord_dim, config.num_frequencies);
    const int64_t expected_input_dim = cicfmInputDim(
        config.coord_dim, config.value_dim, config.num_frequencies);
    if (features.shape()[2] != expected_feature_dim ||
        model.config.input_dim != expected_input_dim) {
      throw std::invalid_argument("CICFM input dimension mismatch");
    }

    CicfmBatch cicfm = makeConfiguredCicfmBatch(features, targets, config);
    model_input = cicfm.input;
    loss_target = cicfm.target_velocity;
  } else {
    throw std::invalid_argument("unknown loss mode");
  }

  if (model.config.output_dim != config.value_dim) {
    throw std::invalid_argument("model output dimension mismatch");
  }

  zeroGrad(model);
  MlpForwardResult forward = mlpForward(model, model_input);
  MseLossResult loss = mseLossAndGradient(forward.output, loss_target);
  TensorStats output_stats = tensorStats(forward.output);
  TensorStats gradient_stats = tensorStats(loss.output_gradient);
  (void)mlpBackward(model, forward.cache, loss.output_gradient);

  if (!output_stats.all_finite || !gradient_stats.all_finite ||
      !allGradientsFinite(model)) {
    throw std::runtime_error("non-finite train-step output or gradient");
  }

  sgdStep(model, config.learning_rate);
  if (!allParametersFinite(model)) {
    throw std::runtime_error("non-finite train-step parameter");
  }

  return TrainStepResult{loss.loss,     model_input,          forward.output,
                         loss_target,   loss.output_gradient, output_stats,
                         gradient_stats};
}
