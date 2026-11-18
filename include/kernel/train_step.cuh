#pragma once
#include "gpu_mlp.cuh"
#include "training/loss.h"

struct GpuTrainContext {
  GpuMlp mlp;
  GpuForwardCache cache;
  float *d_features; // [rows, feature_dim]
  float *d_targets;  // [rows, channels]
  float *d_z0;       // uploaded per step (cicfm input) starting point
  float *d_t;        // uploaded per step (cicfm input) time/clock
  float *d_zt;       // kernel assembly (cicfm output)  current position
  float *d_velocity; // kernel assembly (cicfm output)
  float *d_loss;     // [1]
  int *d_status;     // validation flag; copied back before dependent work
  int rows;
  int feature_dim;
  int channels;
};

GpuTrainContext createGpuTrainContext(const Mlp &model, const MlpConfig &config,
                                      int rows, int feature_dim, int channels);

void freeGpuTrainContext(GpuTrainContext &ctx);

struct StageTimings {
  float assembly_ms; // CICFM only; 0 for MSE
  float forward_ms;
  float loss_ms;     // mse_loss + mse_grad kernels
  float backward_ms; // gpuMlpBackward
  float step_ms;     // gpuTrainStep
};

float gpuTrainStep(GpuTrainContext &ctx, LossMode mode, const Tensor &features,
                   const Tensor &targets, const Tensor *z0, const Tensor *t,
                   float lr, StageTimings *timings = nullptr);

// Views remain valid until the next step or context destruction. For MSE,
// loss_target borrows the caller's targets and has the same lifetime requirement.
struct GpuTrainStepResult {
  const float *prediction; // pre-SGD output, [rows, channels]
  const float *loss_target; // RGB for MSE, velocity for CICFM
  const float *loss_sum; // device scalar; divide by rows*channels for mean loss
};

// Inputs are contiguous device allocations on the current device, sized from ctx:
// features [rows, feature_dim], targets/z0 [rows, channels], t [rows].
// Caller retains ownership and must keep inputs alive through result consumption.
// Do not modify context metadata or alias its model/cache with caller inputs.
// Default stream only. Throws on invalid metadata/data; abort the run on failure.
// Performs small status readbacks, but no input uploads or loss download.
GpuTrainStepResult gpuTrainStepDevice(GpuTrainContext &ctx, LossMode mode,
    const float *features, const float *targets, const float *z0, const float *t,
    float lr, StageTimings *timings = nullptr);
