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
  int rows;
  int feature_dim;
  int channels;
};

GpuTrainContext createGpuTrainContext(const Mlp &model, const MlpConfig &config,
                                      int rows, int feature_dim, int channels);

void freeGpuTrainContext(GpuTrainContext &ctx);

float gpuTrainStep(GpuTrainContext &ctx, LossMode mode, const Tensor &features,
                   const Tensor &targets, const Tensor *z0, const Tensor *t,
                   float lr);
