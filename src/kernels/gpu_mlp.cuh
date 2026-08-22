#pragma once
#include "cuda_utils.cuh"
#include "model/mlp.h"

struct GpuLinearLayer {
  float *weight;
  float *bias;
  float *grad_weight;
  float *grad_bias;
  int in_dim;
  int out_dim;
};

struct GpuMlp {
  GpuLinearLayer layers[4]; // 3 hidden + 1 output
};

inline GpuMlp uploadMlp(const Mlp &mlp) {
  GpuMlp gpu_mlp;
  const int n = static_cast<int>(mlp.layers.size());
  for (int i = 0; i < n; i++) {
    const LinearLayer &src = mlp.layers[i];
    GpuLinearLayer layer;
    layer.out_dim = static_cast<int>(src.weight.shape()[0]);
    layer.in_dim = static_cast<int>(src.weight.shape()[1]);

    const size_t w_count = static_cast<size_t>(layer.in_dim) * layer.out_dim;
    const size_t b_count = static_cast<size_t>(layer.out_dim);

    layer.weight = cuda_alloc(w_count);
    cuda_h2d(layer.weight, src.weight.data(), w_count);
    layer.bias = cuda_alloc(b_count);
    cuda_h2d(layer.bias, src.bias.data(), b_count);

    layer.grad_weight = cuda_alloc(w_count);
    CUDA_CHECK(cudaMemset(layer.grad_weight, 0, w_count * sizeof(float)));
    layer.grad_bias = cuda_alloc(b_count);
    CUDA_CHECK(cudaMemset(layer.grad_bias, 0, b_count * sizeof(float)));

    gpu_mlp.layers[i] = layer;
  }
  return gpu_mlp;
}

inline void downloadMlp(const GpuMlp &gpu, Mlp &mlp) {
  const int n = static_cast<int>(mlp.layers.size());
  for (int i = 0; i < n; i++) {
    const GpuLinearLayer &layer = gpu.layers[i];
    LinearLayer &dst = mlp.layers[i];
    const size_t w_count = static_cast<size_t>(layer.in_dim) * layer.out_dim;
    const size_t b_count = static_cast<size_t>(layer.out_dim);

    cuda_d2h(dst.weight.data(), layer.weight, w_count);
    cuda_d2h(dst.bias.data(), layer.bias, b_count);
    cuda_d2h(dst.grad_weight.data(), layer.grad_weight, w_count);
    cuda_d2h(dst.grad_bias.data(), layer.grad_bias, b_count);
  }
}

inline void freeGpuMlp(GpuMlp &gpu) {
  int n = sizeof(gpu.layers) / sizeof(gpu.layers[0]);
  for (int i = 0; i < n; i++) {
    cudaFree(gpu.layers[i].weight);
    cudaFree(gpu.layers[i].bias);
    cudaFree(gpu.layers[i].grad_weight);
    cudaFree(gpu.layers[i].grad_bias);
  }
}

struct GpuForwardCache {
  float *input; // [row, input_dim]
  float *pre_activation[3];
  float *activation[3];
  float *output; // [row, output_dim]
  int rows;
};

inline GpuForwardCache allocForwardCache(const MlpConfig &config, int rows) {
  GpuForwardCache cache;
  cache.rows = rows;
  cache.input = cuda_alloc(rows * config.input_dim);
  for (int i = 0; i < 3; i++) {
    cache.pre_activation[i] = cuda_alloc(rows * config.hidden_width);
    cache.activation[i] = cuda_alloc(rows * config.hidden_width);
  }
  cache.output = cuda_alloc(rows * config.output_dim);
  return cache;
}

inline void freeForwardCache(GpuForwardCache &cache) {
  cudaFree(cache.input);
  for (int i = 0; i < 3; i++) {
    cudaFree(cache.pre_activation[i]);
    cudaFree(cache.activation[i]);
  }
  cudaFree(cache.output);
}