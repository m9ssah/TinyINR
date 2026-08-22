#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include "training/loss.h"

#include "cuda_utils.cuh"
#include "mse.cuh"

static void run_case(const std::vector<float> &output,
                     const std::vector<float> &target, const char *name) {
  assert(output.size() == target.size());
  const int n = static_cast<int>(output.size());

  Tensor cpu_output({n});
  Tensor cpu_target({n});
  std::copy(output.begin(), output.end(), cpu_output.data());
  std::copy(target.begin(), target.end(), cpu_target.data());

  const float cpu_loss = mseLoss(cpu_output, cpu_target);
  const Tensor cpu_grad = mseOutputGradient(cpu_output, cpu_target);

  float *d_output = cuda_alloc(n);
  float *d_target = cuda_alloc(n);
  float *d_loss = cuda_alloc(1);
  float *d_grad = cuda_alloc(n);
  cuda_h2d(d_output, output.data(), n);
  cuda_h2d(d_target, target.data(), n);
  CUDA_CHECK(cudaMemset(d_loss, 0, sizeof(float)));

  const int blocks = compute_grid_size(n);
  mse_loss<<<blocks, THREADS_PER_BLOCK>>>(d_output, d_target, d_loss, n);
  CUDA_CHECK_LAST_ERROR();
  mse_grad<<<blocks, THREADS_PER_BLOCK>>>(d_output, d_target, d_grad, n);
  CUDA_CHECK_LAST_ERROR();
  CUDA_CHECK(cudaDeviceSynchronize());

  float loss_sum = 0.0f;
  std::vector<float> gpu_grad(n);
  cuda_d2h(&loss_sum, d_loss, 1);
  cuda_d2h(gpu_grad.data(), d_grad, n);
  const float gpu_loss = loss_sum / static_cast<float>(n);

  std::cout << name << " loss: cpu=" << cpu_loss << " gpu=" << gpu_loss << "\n";
  assert(std::fabs(cpu_loss - gpu_loss) <= 1e-5f * std::fabs(cpu_loss) + 1e-7f);

  std::cout << name << " grad: ";
  assert(check_parity_rel(cpu_grad.data(), gpu_grad.data(), n, 1e-4f, 1e-6f));

  CUDA_CHECK(cudaFree(d_output));
  CUDA_CHECK(cudaFree(d_target));
  CUDA_CHECK(cudaFree(d_loss));
  CUDA_CHECK(cudaFree(d_grad));
}

int main() {
  run_case({1, 2, 3, 4, 5, 6}, {2, 4, 3, 1, 5, 0}, "regular");

  std::mt19937 rng(123);
  std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
  std::vector<float> random_output(10000);
  std::vector<float> random_target(10000);
  for (float &x : random_output) {
    x = dist(rng);
  }
  for (float &x : random_target) {
    x = dist(rng);
  }
  run_case(random_output, random_target, "random");

  std::cout << "test_mse_cuda PASS\n";
  return 0;
}
