#include "kernel/adamw.cuh"

#include <cmath>

#include "kernel/cuda_utils.cuh"

static __device__ __inline__ float adamwUpdate(float param, float grad,
                                               float *m, float *v,
                                               const AdamWStepParams &p,
                                               float decay) {
  const float m_new = p.beta1 * (*m) + (1.0f - p.beta1) * grad;
  const float v_new = p.beta2 * (*v) + (1.0f - p.beta2) * grad * grad;
  *m = m_new;
  *v = v_new;
  return param * decay - p.lr * (m_new / p.bias_correction1) /
                             (sqrtf(v_new / p.bias_correction2) + p.eps);
}

__global__ void adam_weight_kernel(float *d_w, float *d_w_T,
                                   const float *d_grad, float *d_m, float *d_v,
                                   AdamWStepParams p, int in_dim, int out_dim) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= in_dim * out_dim)
    return;

  const float w =
      adamwUpdate(d_w[idx], d_grad[idx], &d_m[idx], &d_v[idx], p, p.decay);
  d_w[idx] = w;
  const int o = idx / in_dim;
  const int i = idx % in_dim;
  d_w_T[i * out_dim + o] = w;
}

__global__ void adam_bias_kernel(float *d_b, const float *d_grad, float *d_m,
                                 float *d_v, AdamWStepParams p, int n) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n)
    return;
  d_b[idx] = adamwUpdate(d_b[idx], d_grad[idx], &d_m[idx], &d_v[idx], p, 1.0f);
}

void gpuAdamWStep(GpuMlp &mlp, GpuAdamW &adamw, const AdamWConfig &cfg) {
  adamw.step++;

  AdamWStepParams p;
  p.lr = cfg.lr;
  p.beta1 = cfg.beta1;
  p.beta2 = cfg.beta2;
  p.eps = cfg.eps;
  p.bias_correction1 = static_cast<float>(
      1.0 - std::pow(static_cast<double>(cfg.beta1), adamw.step));
  p.bias_correction2 = static_cast<float>(
      1.0 - std::pow(static_cast<double>(cfg.beta2), adamw.step));
  p.decay = 1.0f - cfg.lr * cfg.weight_decay;

  const int n = sizeof(mlp.layers) / sizeof(mlp.layers[0]);
  for (int i = 0; i < n; i++) {
    GpuLinearLayer &layer = mlp.layers[i];
    const GpuAdamWLayer &state = adamw.layers[i];
    const int w_count = layer.in_dim * layer.out_dim;
    const int b_count = layer.out_dim;

    adam_weight_kernel<<<compute_grid_size(w_count), THREADS_PER_BLOCK>>>(
        layer.weight, layer.weight_T, layer.grad_weight, state.m_w, state.v_w,
        p, layer.in_dim, layer.out_dim);
    CUDA_CHECK_LAST_ERROR();

    adam_bias_kernel<<<compute_grid_size(b_count), THREADS_PER_BLOCK>>>(
        layer.bias, layer.grad_bias, state.m_b, state.v_b, p, b_count);
    CUDA_CHECK_LAST_ERROR();
  }
}
