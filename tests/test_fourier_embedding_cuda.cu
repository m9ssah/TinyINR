#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

#include "../src/kernels/coordinate_embedding.cuh"
#include "../src/kernels/cuda_utils.cuh"


#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static float expected(float x, int f, int trig) {
  float angle = powf(2.0f, f) * static_cast<float>(M_PI) * x;
  return trig == 0 ? sinf(angle) : cosf(angle);
}

int main() {
  const int B = 1;
  const int N = 2;
  const int D = 2;
  const int F = 2;

  std::vector<float> input = {0.0f, 1.0f, -1.0f, 0.5f};

  std::vector<float> output(B * N * D * F * 2, 0.0f);

  float *d_input = cuda_alloc(input.size());
  float *d_output = cuda_alloc(output.size());

  cuda_h2d(d_input, input.data(), input.size());

  int total = B * N * D;
  int threads = THREADS_PER_BLOCK;
  int blocks = compute_grid_size(total, threads);

  coordinate_embedding<<<blocks, threads>>>(d_input, d_output, B, N, D, F);
  CUDA_CHECK_LAST_ERROR();
  CUDA_CHECK(cudaDeviceSynchronize());

  cuda_d2h(output.data(), d_output, output.size());

  for (int idx = 0; idx < total; ++idx) {
    for (int f = 0; f < F; ++f) {
      for (int trig = 0; trig < 2; ++trig) {
        int out_idx = idx * F * 2 + f * 2 + trig;
        float diff = fabsf(output[out_idx] - expected(input[idx], f, trig));
        assert(diff < 1e-5f);
      }
    }
  }

  CUDA_CHECK(cudaFree(d_input));
  CUDA_CHECK(cudaFree(d_output));

  std::cout << "test_fourier_embedding_cuda PASS\n";
  return 0;
}