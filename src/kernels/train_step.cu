#include "kernel/train_step.cuh"

#include <cassert>
#include <stdexcept>

#include "kernel/cicfm.cuh"
#include "kernel/cuda_utils.cuh"
#include "kernel/mlp_backward.cuh"
#include "kernel/mlp_forward.cuh"
#include "kernel/mse.cuh"
#include "optim/sgd.cuh"

GpuTrainContext createGpuTrainContext(const Mlp &model, const MlpConfig &config,
                                      int rows, int feature_dim, int channels) {
  GpuTrainContext ctx;
  ctx.mlp = uploadMlp(model);
  ctx.cache = allocForwardCache(config, rows);
  ctx.d_features = cuda_alloc(static_cast<size_t>(rows) * feature_dim);
  ctx.d_targets = cuda_alloc(static_cast<size_t>(rows) * channels);
  ctx.d_z0 = cuda_alloc(static_cast<size_t>(rows) * channels);
  ctx.d_t = cuda_alloc(static_cast<size_t>(rows));
  ctx.d_zt = cuda_alloc(static_cast<size_t>(rows) * channels);
  ctx.d_velocity = cuda_alloc(static_cast<size_t>(rows) * channels);
  ctx.d_loss = cuda_alloc(1);
  ctx.rows = rows;
  ctx.feature_dim = feature_dim;
  ctx.channels = channels;
  return ctx;
}

void freeGpuTrainContext(GpuTrainContext &ctx) {
  freeGpuMlp(ctx.mlp);
  freeForwardCache(ctx.cache);
  cudaFree(ctx.d_features);
  cudaFree(ctx.d_targets);
  cudaFree(ctx.d_z0);
  cudaFree(ctx.d_t);
  cudaFree(ctx.d_zt);
  cudaFree(ctx.d_velocity);
  cudaFree(ctx.d_loss);
}

// shared part of training: zero grads, forward, MSE loss, and gradient against
// loss_target, backward. data movement & optimizer are what set gpuTrainStep
// and gpuTrainStepResident apart.
static void forwardLossBackward(GpuTrainContext &ctx,
                                const float *loss_target) {
  gpuZeroGrad(ctx.mlp);
  gpuMlpForward(ctx.mlp, ctx.cache);
  gpuMseLossAndGrad(ctx.cache.output, loss_target, ctx.d_loss,
                    ctx.cache.grad_output, ctx.rows * ctx.channels);
  gpuMlpBackward(ctx.mlp, ctx.cache);
}

float gpuTrainStep(GpuTrainContext &ctx, LossMode mode, const Tensor &features,
                   const Tensor &targets, const Tensor *z0, const Tensor *t,
                   float lr) { // TODO: will retire this soon
  const int rows = ctx.rows;
  const int out_count = rows * ctx.channels;

  // choose the model input and the loss target
  const float *loss_target;
  if (mode == LossMode::CICFM) {
    assert(z0 != nullptr && t != nullptr);
    cuda_h2d(ctx.d_targets, targets.data(), static_cast<size_t>(out_count));
    cuda_h2d(ctx.d_features, features.data(),
             static_cast<size_t>(rows) * ctx.feature_dim);
    cuda_h2d(ctx.d_z0, z0->data(), static_cast<size_t>(out_count));
    cuda_h2d(ctx.d_t, t->data(), static_cast<size_t>(rows));
    gpuCicfmAssembly(ctx.d_features, ctx.d_z0, ctx.d_targets, ctx.d_t, ctx.d_zt,
                     ctx.d_velocity, ctx.cache.input, rows, ctx.feature_dim,
                     ctx.channels);
    loss_target = ctx.d_velocity;
  } else {
    assert(z0 == nullptr && t == nullptr);
    gpuUploadTrainData(ctx, features, targets);
    loss_target = ctx.d_targets;
  }

  forwardLossBackward(ctx, loss_target);
  gpuSgdStep(ctx.mlp, lr);
  return gpuReadLoss(ctx);
}

void gpuUploadTrainData(GpuTrainContext &ctx, const Tensor &features,
                        const Tensor &targets) {
  const size_t feature_count = static_cast<size_t>(ctx.rows) * ctx.feature_dim;
  const size_t target_count = static_cast<size_t>(ctx.rows) * ctx.channels;

  if (ctx.mlp.layers[0].in_dim != ctx.feature_dim) {
    throw std::invalid_argument(
        "gpuUploadTrainData: model input width != feature_dim (MSE only)");
  }
  if (static_cast<size_t>(features.numel()) != feature_count ||
      static_cast<size_t>(targets.numel()) != target_count) {
    throw std::invalid_argument(
        "gpuUploadTrainData: features/targets don't match context shape");
  }

  cuda_h2d(ctx.cache.input, features.data(), feature_count);
  cuda_h2d(ctx.d_targets, targets.data(), target_count);
}

void gpuTrainStepResident(GpuTrainContext &ctx, GpuAdamW &adamw,
                          const AdamWConfig &cfg) {
  forwardLossBackward(ctx, ctx.d_targets);
  gpuAdamWStep(ctx.mlp, adamw, cfg);
}

float gpuReadLoss(const GpuTrainContext &ctx) {
  float loss_sum = 0.0f;
  cuda_d2h(&loss_sum, ctx.d_loss, 1);
  return loss_sum / static_cast<float>(ctx.rows * ctx.channels);
}