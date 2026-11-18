#include "kernel/train_step.cuh"

#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>

#include "kernel/cicfm.cuh"
#include "kernel/cuda_utils.cuh"
#include "kernel/linear.cuh"
#include "kernel/mlp_backward.cuh"
#include "kernel/mlp_forward.cuh"
#include "kernel/mse.cuh"
#include "kernel/sgd.cuh"

namespace {

void require(bool condition, const char *message) {
  if (!condition) throw std::invalid_argument(message);
}

void checkCount(int64_t a, int64_t b) {
  // Existing kernels use signed int offsets and round up launch dimensions.
  const int64_t limit = std::numeric_limits<int>::max() - THREADS_PER_BLOCK;
  require(a > 0 && b > 0 && a <= limit / b,
          "GPU training size exceeds signed-int kernel limits");
}

void validateModel(const Mlp &model, const MlpConfig &config,
                   int rows, int features, int channels) {
  require(config.hidden_layer_count == 3 && model.layers.size() == 4 &&
          config.activation == ActivationKind::SiLU,
          "GPU training requires exactly three SiLU hidden layers");
  const auto &actual = model.config;
  require(actual.input_dim == config.input_dim &&
          actual.output_dim == config.output_dim &&
          actual.hidden_width == config.hidden_width &&
          actual.hidden_layer_count == config.hidden_layer_count &&
          actual.activation == config.activation, "GPU model/config mismatch");
  checkCount(rows, features);
  checkCount(rows, channels);
  checkCount(rows, config.input_dim);
  checkCount(rows, config.hidden_width);
  require(rows <= 65535 * TILE_SIZE, "GPU training exceeds tiled grid-y limit");
  require(config.output_dim == channels &&
          (config.input_dim == features ||
           config.input_dim == int64_t(features) + channels + 1),
          "GPU model input/output dimensions do not match training");
  for (int i = 0; i < 4; ++i) {
    const int64_t in = i == 0 ? config.input_dim : config.hidden_width;
    const int64_t out = i == 3 ? config.output_dim : config.hidden_width;
    checkCount(in, out);
    const auto &layer = model.layers[i];
    require(layer.weight.shape() == std::vector<int64_t>({out, in}) &&
            layer.bias.shape() == std::vector<int64_t>({out}) &&
            layer.grad_weight.shape() == layer.weight.shape() &&
            layer.grad_bias.shape() == layer.bias.shape(),
            "GPU model layer shape mismatch");
    for (const Tensor *tensor : {&layer.weight, &layer.bias})
      for (int64_t j = 0; j < tensor->numel(); ++j)
        require(std::isfinite(tensor->data()[j]), "GPU model must be finite");
  }
}

void validateStep(const GpuTrainContext &ctx, LossMode mode, float lr,
                  bool has_z0, bool has_t) {
  require(ctx.d_status != nullptr, "GPU training context is not live");
  require(mode == LossMode::MSE || mode == LossMode::CICFM,
          "Unknown GPU loss mode");
  require(std::isfinite(lr) && lr > 0, "Learning rate must be finite and positive");
  const bool cicfm = mode == LossMode::CICFM;
  require(has_z0 == cicfm && has_t == cicfm,
          "CICFM requires z0 and t; MSE requires neither");
  require(ctx.mlp.layers[0].in_dim == int64_t(ctx.feature_dim) +
          (cicfm ? int64_t(ctx.channels) + 1 : 0),
          "GPU model input width does not match loss mode");
}

void requireDevicePointer(const float *ptr) {
  require(ptr != nullptr, "GPU input pointer must not be null");
  cudaPointerAttributes attr{};
  const cudaError_t error = cudaPointerGetAttributes(&attr, ptr);
  if (error != cudaSuccess) {
    cudaGetLastError();
    throw std::invalid_argument("GPU input pointer is not a device allocation");
  }
  int device;
  CUDA_CHECK(cudaGetDevice(&device));
  require(attr.type == cudaMemoryTypeDevice && attr.device == device,
          "GPU input must be a device allocation on the current device");
}

__global__ void validateValues(const float *values, int count, int *status,
                               bool time = false) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count && (!isfinite(values[i]) ||
                   (time && (values[i] < 0 || values[i] > 1))))
    atomicExch(status, 1);
}

__global__ void validateUpdate(const float *param, const float *grad, int count,
                               float lr, int *status) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count && (!isfinite(param[i]) || !isfinite(grad[i]) ||
                   !isfinite(param[i] - lr * grad[i])))
    atomicExch(status, 1);
}

void checkValues(GpuTrainContext &ctx, const float *values, int count,
                 bool time = false) {
  validateValues<<<compute_grid_size(count), THREADS_PER_BLOCK>>>(
      values, count, ctx.d_status, time);
  CUDA_CHECK_LAST_ERROR();
}

void requireValidStatus(GpuTrainContext &ctx) {
  int status;
  CUDA_CHECK(cudaMemcpy(&status, ctx.d_status, sizeof(status), cudaMemcpyDeviceToHost));
  if (status) throw std::runtime_error("Invalid or nonfinite GPU training values; SGD not applied");
}

void validateHostInputs(const GpuTrainContext &ctx, const Tensor &features,
                        const Tensor &targets, const Tensor *z0, const Tensor *t) {
  require((features.ndim() == 2 || features.ndim() == 3) &&
          features.shape().back() == ctx.feature_dim &&
          features.numel() == int64_t(ctx.rows) * ctx.feature_dim,
          "GPU training feature shape mismatch");
  auto shape = features.shape();
  shape.back() = ctx.channels;
  require(targets.shape() == shape, "GPU training target shape mismatch");
  if (z0) require(z0->shape() == shape, "GPU training z0 shape mismatch");
  shape.back() = 1;
  if (t) require(t->shape() == shape, "GPU training time shape mismatch");
}

} // namespace

GpuTrainContext createGpuTrainContext(const Mlp &model, const MlpConfig &config,
                                      int rows, int feature_dim, int channels) {
  validateModel(model, config, rows, feature_dim, channels);
  GpuTrainContext ctx{};
  ctx.mlp = uploadMlp(model);
  ctx.cache = allocForwardCache(config, rows);
  ctx.d_features = cuda_alloc(static_cast<size_t>(rows) * feature_dim);
  ctx.d_targets = cuda_alloc(static_cast<size_t>(rows) * channels);
  ctx.d_z0 = cuda_alloc(static_cast<size_t>(rows) * channels);
  ctx.d_t = cuda_alloc(static_cast<size_t>(rows));
  ctx.d_zt = cuda_alloc(static_cast<size_t>(rows) * channels);
  ctx.d_velocity = cuda_alloc(static_cast<size_t>(rows) * channels);
  ctx.d_loss = cuda_alloc(1);
  CUDA_CHECK(cudaMalloc(&ctx.d_status, sizeof(int)));
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
  cudaFree(ctx.d_status);
  ctx = GpuTrainContext{};
}

float gpuTrainStep(GpuTrainContext &ctx, LossMode mode, const Tensor &features,
                   const Tensor &targets, const Tensor *z0, const Tensor *t,
                   float lr, StageTimings *timings) {
  validateStep(ctx, mode, lr, z0 != nullptr, t != nullptr);
  validateHostInputs(ctx, features, targets, z0, t);
  std::unique_ptr<GpuTimer> timer;
  if (timings) { timer.reset(new GpuTimer); timer->start(); }
  cuda_h2d(ctx.d_features, features.data(), features.numel());
  cuda_h2d(ctx.d_targets, targets.data(), targets.numel());
  if (z0) cuda_h2d(ctx.d_z0, z0->data(), z0->numel());
  if (t) cuda_h2d(ctx.d_t, t->data(), t->numel());
  const auto result = gpuTrainStepDevice(ctx, mode, ctx.d_features, ctx.d_targets,
      z0 ? ctx.d_z0 : nullptr, t ? ctx.d_t : nullptr, lr, timings);
  float loss;
  cuda_d2h(&loss, result.loss_sum, 1);
  if (timings) { timer->stop(); timings->step_ms = timer->elapsed_ms(); }
  return loss / static_cast<float>(ctx.rows * ctx.channels);
}

GpuTrainStepResult gpuTrainStepDevice(GpuTrainContext &ctx, LossMode mode,
    const float *features, const float *targets, const float *z0, const float *t,
    float lr, StageTimings *timings) {
  validateStep(ctx, mode, lr, z0 != nullptr, t != nullptr);
  requireDevicePointer(ctx.cache.input);
  requireDevicePointer(features);
  requireDevicePointer(targets);
  if (z0) requireDevicePointer(z0);
  if (t) requireDevicePointer(t);
  const int rows = ctx.rows;
  const int out_count = rows * ctx.channels;
  std::unique_ptr<GpuTimer> step_timer, stage_timer;
  if (timings) {
    *timings = StageTimings{};
    step_timer.reset(new GpuTimer);
    stage_timer.reset(new GpuTimer);
    step_timer->start();
  }

  CUDA_CHECK(cudaMemset(ctx.d_status, 0, sizeof(int)));
  checkValues(ctx, features, rows * ctx.feature_dim);
  checkValues(ctx, targets, out_count);
  if (z0) checkValues(ctx, z0, out_count);
  if (t) checkValues(ctx, t, rows, true);
  requireValidStatus(ctx);

  const float *loss_target = targets;
  if (mode == LossMode::CICFM) {
    if (timings) stage_timer->start();
    cicfm_assembly_kernel<<<compute_grid_size(rows), THREADS_PER_BLOCK>>>(
        features, z0, targets, t, ctx.d_zt, ctx.d_velocity, ctx.cache.input,
        rows, ctx.feature_dim, ctx.channels);
    CUDA_CHECK_LAST_ERROR();
    if (timings) {
      stage_timer->stop();
      timings->assembly_ms = stage_timer->elapsed_ms();
    }
    loss_target = ctx.d_velocity;
    checkValues(ctx, ctx.cache.input, rows * ctx.mlp.layers[0].in_dim);
    checkValues(ctx, loss_target, out_count);
    requireValidStatus(ctx);
  } else {
    CUDA_CHECK(cudaMemcpyAsync(ctx.cache.input, features,
        size_t(rows) * ctx.feature_dim * sizeof(float), cudaMemcpyDeviceToDevice));
  }
  gpuZeroGrad(ctx.mlp);

  if (timings) stage_timer->start();
  gpuMlpForward(ctx.mlp, ctx.cache);
  if (timings) {
    stage_timer->stop();
    timings->forward_ms = stage_timer->elapsed_ms();
    stage_timer->start();
  }

  CUDA_CHECK(cudaMemset(ctx.d_loss, 0, sizeof(float)));
  mse_loss_kernel<<<compute_grid_size(out_count), THREADS_PER_BLOCK>>>(
      ctx.cache.output, loss_target, ctx.d_loss, out_count);
  CUDA_CHECK_LAST_ERROR();
  mse_grad_kernel<<<compute_grid_size(out_count), THREADS_PER_BLOCK>>>(
      ctx.cache.output, loss_target, ctx.cache.grad_output, out_count);
  CUDA_CHECK_LAST_ERROR();
  if (timings) {
    stage_timer->stop();
    timings->loss_ms = stage_timer->elapsed_ms();
    stage_timer->start();
  }

  gpuMlpBackward(ctx.mlp, ctx.cache);
  if (timings) {
    stage_timer->stop();
    timings->backward_ms = stage_timer->elapsed_ms();
  }
  checkValues(ctx, ctx.cache.output, out_count);
  checkValues(ctx, ctx.d_loss, 1);
  for (const auto &layer : ctx.mlp.layers) {
    const int count = layer.in_dim * layer.out_dim;
    validateUpdate<<<compute_grid_size(count), THREADS_PER_BLOCK>>>(
        layer.weight, layer.grad_weight, count, lr, ctx.d_status);
    CUDA_CHECK_LAST_ERROR();
    validateUpdate<<<compute_grid_size(layer.out_dim), THREADS_PER_BLOCK>>>(
        layer.bias, layer.grad_bias, layer.out_dim, lr, ctx.d_status);
    CUDA_CHECK_LAST_ERROR();
  }
  // All layers must validate before any parameter is updated.
  requireValidStatus(ctx);
  gpuSgdStep(ctx.mlp, lr);
  if (timings) {
    step_timer->stop();
    timings->step_ms = step_timer->elapsed_ms();
  }
  return {ctx.cache.output, loss_target, ctx.d_loss};
}
