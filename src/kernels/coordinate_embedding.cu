#include "coordinate_embedding.cuh"
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

__global__ void coordinate_embedding(const float *d_input, float *d_output,
                                     int B, int N, int D, int F) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = B * N * D;

  if (idx < total) {
    float x = d_input[idx];
    for (int f = 0; f < F; f++) {
      float freq = powf(2.0f, f) * M_PI;
      d_output[idx * F * 2 + f * 2] = sinf(freq * x);
      d_output[idx * F * 2 + f * 2 + 1] = cosf(freq * x);
    }
  }
}

// CPU ref:
void cpu_coordinate_embedding(const float *input, float *output, int B, int N,
                              int D, int F) {
  int total = B * N * D;
  for (int idx = 0; idx < total; idx++) {
    float x = input[idx];
    for (int f = 0; f < F; f++) {
      float freq = powf(2.0f, f) * M_PI;
      output[idx * F * 2 + f * 2 + 0] = sinf(freq * x);
      output[idx * F * 2 + f * 2 + 1] = cosf(freq * x);
    }
  }
}