#pragma once

#include "kernel/gpu_mlp.cuh"

struct AdamWConfig {
  float lr = 1e-3f;
  float beta1 = 0.9f;
  float beta2 = 0.999f;
  float eps = 1e-8f;
  float weight_decay = 0.0f;
};

struct GpuAdamWLayer {
  float *m_w;
  float *v_w;
  float *m_b;
  float *v_b;
};

struct GpuAdamW {
  GpuAdamWLayer layers[4];
  int step = 0;
};

GpuAdamW create_gpu_adamw(GpuMlp &mlp);

void free_gpu_adamw(GpuAdamW &gpu_adamw);

void gpu_adamw_step(GpuAdamW &gpu_adamw, const GpuMlp &mlp,
                    const AdamWConfig &cfg);
