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
  float *m_w = nullptr;
  float *v_w = nullptr;
  float *m_b = nullptr;
  float *v_b = nullptr;
};

struct GpuAdamW {
  GpuAdamWLayer layers[4];
  int step = 0;
};

struct AdamWStepParams {
  float lr;
  float beta1;
  float beta2;
  float eps;
  float bias_correction1;
  float bias_correction2;
  float decay;
};

__global__ void adam_weight_kernel(float *d_w, float *d_w_T,
                                   const float *d_grad, float *d_m, float *d_v,
                                   AdamWStepParams p, int in_dim, int out_dim);

__global__ void adam_bias_kernel(float *d_b, const float *d_grad, float *d_m,
                                 float *d_v, AdamWStepParams p, int n);

void gpuAdamWStep(GpuMlp &mlp, GpuAdamW &adamw, const AdamWConfig &cfg);

inline GpuAdamW createGpuAdamW(const GpuMlp &mlp) {
  GpuAdamW adamw;
  const int n = sizeof(mlp.layers) / sizeof(mlp.layers[0]);
  for (int i = 0; i < n; i++) {
    const GpuLinearLayer &layer = mlp.layers[i];
    const size_t w_count = static_cast<size_t>(layer.in_dim) * layer.out_dim;
    const size_t b_count = static_cast<size_t>(layer.out_dim);
    GpuAdamWLayer &state = adamw.layers[i];

    state.m_w = cuda_alloc(w_count);
    state.v_w = cuda_alloc(w_count);
    state.m_b = cuda_alloc(b_count);
    state.v_b = cuda_alloc(b_count);
    
    // need to memset to 0 because adamwUpdate uses the previous m and v values
    CUDA_CHECK(cudaMemset(state.m_w, 0, w_count * sizeof(float)));
    CUDA_CHECK(cudaMemset(state.v_w, 0, w_count * sizeof(float)));
    CUDA_CHECK(cudaMemset(state.m_b, 0, b_count * sizeof(float)));
    CUDA_CHECK(cudaMemset(state.v_b, 0, b_count * sizeof(float)));
  }
  return adamw;
}

inline void freeGpuAdamW(GpuAdamW &adamw) {
  const int n = sizeof(adamw.layers) / sizeof(adamw.layers[0]);
  for (int i = 0; i < n; i++) {
    cudaFree(adamw.layers[i].m_w);
    cudaFree(adamw.layers[i].v_w);
    cudaFree(adamw.layers[i].m_b);
    cudaFree(adamw.layers[i].v_b);
  }
}
