#include "train_step.cuh"

#include <cassert>

#include "cicfm.cuh"
#include "cuda_utils.cuh"
#include "mlp_backward.cuh"
#include "mlp_forward.cuh"
#include "mse.cuh"
#include "sgd.cuh"

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

float gpuTrainStep(GpuTrainContext &ctx, LossMode mode, const Tensor &features,
                   const Tensor &targets, const Tensor *z0, const Tensor *t,
                   float lr) {
  const int rows = ctx.rows;
  const int out_count = rows * ctx.channels;

  cuda_h2d(ctx.d_targets, targets.data(), static_cast<size_t>(out_count));

  // Mode dispatch: choose the model input and the loss target. Everything
  // after this block is identical for both modes.
  const float *loss_target;
  if (mode == LossMode::CICFM) {
    assert(z0 != nullptr && t != nullptr);
    cuda_h2d(ctx.d_features, features.data(),
             static_cast<size_t>(rows) * ctx.feature_dim);
    cuda_h2d(ctx.d_z0, z0->data(), static_cast<size_t>(out_count));
    cuda_h2d(ctx.d_t, t->data(), static_cast<size_t>(rows));

    cicfm_assembly_kernel<<<compute_grid_size(rows), THREADS_PER_BLOCK>>>(
        ctx.d_features, ctx.d_z0, ctx.d_targets, ctx.d_t, ctx.d_zt,
        ctx.d_velocity, ctx.cache.input, rows, ctx.feature_dim, ctx.channels);
    CUDA_CHECK_LAST_ERROR();

    loss_target = ctx.d_velocity;
  } else {
    assert(z0 == nullptr && t == nullptr);
    cuda_h2d(ctx.cache.input, features.data(),
             static_cast<size_t>(rows) * ctx.feature_dim);
    loss_target = ctx.d_targets;
  }

  gpuZeroGrad(ctx.mlp);
  gpuMlpForward(ctx.mlp, ctx.cache);

  CUDA_CHECK(cudaMemset(ctx.d_loss, 0, sizeof(float)));
  mse_loss_kernel<<<compute_grid_size(out_count), THREADS_PER_BLOCK>>>(
      ctx.cache.output, loss_target, ctx.d_loss, out_count);
  CUDA_CHECK_LAST_ERROR();
  mse_grad_kernel<<<compute_grid_size(out_count), THREADS_PER_BLOCK>>>(
      ctx.cache.output, loss_target, ctx.cache.grad_output, out_count);
  CUDA_CHECK_LAST_ERROR();

  gpuMlpBackward(ctx.mlp, ctx.cache);
  gpuSgdStep(ctx.mlp, lr);

  float loss_sum = 0.0f;
  cuda_d2h(&loss_sum, ctx.d_loss, 1);
  return loss_sum / static_cast<float>(out_count);
}
