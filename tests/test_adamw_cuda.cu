#undef NDEBUG
#include <cassert>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include "model/mlp.h"

#include "kernel/adamw.cuh"
#include "kernel/cuda_utils.cuh"

namespace {

MlpConfig tinyConfig() {
  MlpConfig config;
  config.input_dim = 5;
  config.hidden_width = 7;
  config.output_dim = 3;
  config.hidden_layer_count = 3;
  config.activation = ActivationKind::SiLU;
  return config;
}

struct HostParams {
  std::vector<float> w, b, m_w, v_w, m_b, v_b;
  int in_dim = 0, out_dim = 0;
};

std::vector<HostParams> hostCopy(const Mlp &model) {
  std::vector<HostParams> params(4);
  for (int i = 0; i < 4; i++) {
    const LinearLayer &layer = model.layers[i];
    HostParams &p = params[i];
    p.out_dim = static_cast<int>(layer.weight.shape()[0]);
    p.in_dim = static_cast<int>(layer.weight.shape()[1]);
    p.w.assign(layer.weight.data(), layer.weight.data() + layer.weight.numel());
    p.b.assign(layer.bias.data(), layer.bias.data() + layer.bias.numel());
    p.m_w.assign(p.w.size(), 0.0f);
    p.v_w.assign(p.w.size(), 0.0f);
    p.m_b.assign(p.b.size(), 0.0f);
    p.v_b.assign(p.b.size(), 0.0f);
  }
  return params;
}

// cpu reference implemntation for parity tests
void cpuAdamW(std::vector<float> &param, const std::vector<float> &grad,
              std::vector<float> &m, std::vector<float> &v,
              const AdamWConfig &cfg, int step, float decay) {
  const float bc1 = static_cast<float>(1.0 - std::pow(double(cfg.beta1), step));
  const float bc2 = static_cast<float>(1.0 - std::pow(double(cfg.beta2), step));
  for (size_t k = 0; k < param.size(); k++) {
    m[k] = cfg.beta1 * m[k] + (1.0f - cfg.beta1) * grad[k];
    v[k] = cfg.beta2 * v[k] + (1.0f - cfg.beta2) * grad[k] * grad[k];
    param[k] = param[k] * decay -
               cfg.lr * (m[k] / bc1) / (std::sqrt(v[k] / bc2) + cfg.eps);
  }
}

std::vector<float> download(const float *d_ptr, size_t count) {
  std::vector<float> host(count);
  cuda_d2h(host.data(), d_ptr, count);
  return host;
}

void checkTransposeExact(const GpuLinearLayer &layer) {
  const size_t count = static_cast<size_t>(layer.in_dim) * layer.out_dim;
  const std::vector<float> w = download(layer.weight, count);
  const std::vector<float> w_t = download(layer.weight_T, count);
  for (int o = 0; o < layer.out_dim; o++)
    for (int i = 0; i < layer.in_dim; i++)
      assert(w_t[static_cast<size_t>(i) * layer.out_dim + o] ==
             w[static_cast<size_t>(o) * layer.in_dim + i]);
}

void assertClose(const std::vector<float> &expected,
                 const std::vector<float> &actual) {
  assert(expected.size() == actual.size());
  for (size_t k = 0; k < expected.size(); k++) {
    const float tol = 1e-6f * std::fabs(expected[k]) + 1e-7f;
    assert(std::fabs(expected[k] - actual[k]) <= tol);
  }
}

void testParity() {
  const MlpConfig config = tinyConfig();
  const Mlp model = createMlp(config, 7);
  GpuMlp gpu = uploadMlp(model);
  GpuAdamW adamw = createGpuAdamW(gpu);
  std::vector<HostParams> ref = hostCopy(model);

  AdamWConfig cfg;
  cfg.lr = 1e-2f;
  cfg.weight_decay = 0.1f;
  const float decay = 1.0f - cfg.lr * cfg.weight_decay;

  std::mt19937 rng(11);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  for (int step = 1; step <= 3; step++) {
    for (int i = 0; i < 4; i++) {
      HostParams &p = ref[i];
      std::vector<float> g_w(p.w.size()), g_b(p.b.size());
      for (float &g : g_w)
        g = dist(rng);
      for (float &g : g_b)
        g = dist(rng);
      cuda_h2d(gpu.layers[i].grad_weight, g_w.data(), g_w.size());
      cuda_h2d(gpu.layers[i].grad_bias, g_b.data(), g_b.size());
      cpuAdamW(p.w, g_w, p.m_w, p.v_w, cfg, step, decay);
      cpuAdamW(p.b, g_b, p.m_b, p.v_b, cfg, step, 1.0f); // biases don't decay
    }
    gpuAdamWStep(gpu, adamw, cfg);

    for (int i = 0; i < 4; i++) {
      assertClose(ref[i].w, download(gpu.layers[i].weight, ref[i].w.size()));
      assertClose(ref[i].b, download(gpu.layers[i].bias, ref[i].b.size()));
      checkTransposeExact(gpu.layers[i]);
    }
  }
  assert(adamw.step == 3);

  freeGpuAdamW(adamw);
  freeGpuMlp(gpu);
  std::cout << "  parity vs CPU reference: ok\n";
}

void testZeroGradientDecay() {
  const MlpConfig config = tinyConfig();
  const Mlp model = createMlp(config, 13);
  GpuMlp gpu = uploadMlp(model);
  GpuAdamW adamw = createGpuAdamW(gpu);
  std::vector<HostParams> ref = hostCopy(model);

  AdamWConfig cfg;
  cfg.lr = 0.1f;
  cfg.weight_decay = 0.5f;
  const float decay = 1.0f - cfg.lr * cfg.weight_decay;

  for (int i = 0; i < 4; i++) {
    const std::vector<float> zeros_w(ref[i].w.size(), 0.0f);
    const std::vector<float> zeros_b(ref[i].b.size(), 0.0f);
    cuda_h2d(gpu.layers[i].grad_weight, zeros_w.data(), zeros_w.size());
    cuda_h2d(gpu.layers[i].grad_bias, zeros_b.data(), zeros_b.size());
  }

  const int steps = 4;
  for (int step = 0; step < steps; step++)
    gpuAdamWStep(gpu, adamw, cfg);

  for (int i = 0; i < 4; i++) {
    std::vector<float> expected = ref[i].w;
    for (float &w : expected)
      for (int step = 0; step < steps; step++)
        w *= decay;
    const std::vector<float> w =
        download(gpu.layers[i].weight, expected.size());
    const std::vector<float> b = download(gpu.layers[i].bias, ref[i].b.size());
    for (size_t k = 0; k < w.size(); k++)
      assert(w[k] == expected[k]);
    for (size_t k = 0; k < b.size(); k++)
      assert(b[k] == ref[i].b[k]);
    checkTransposeExact(gpu.layers[i]);
  }

  freeGpuAdamW(adamw);
  freeGpuMlp(gpu);
  std::cout << "  zero-gradient decay is decoupled and weight-only: ok\n";
}

} // namespace

int main() {
  testParity();
  testZeroGradientDecay();
  std::cout << "test_adamw_cuda PASS\n";
  return 0;
}
