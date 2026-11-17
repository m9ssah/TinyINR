#pragma once

#include <cstddef>
#include <cstdint>

#include "sampling/coordinate_sampler.h"

// Owns all pointers. Do not copy, mutate dimensions, or free individual buffers.
// Default stream/current device only. All allocations occur at creation.
struct GpuCoordinateSampler {
  int B, N, K, D, C;
  SamplingConfig config;
  uint64_t seed;
  float *coordinates, *targets, *priorities;
  int64_t *indices;             // [B,K], local original-point indices
  double *draw_probabilities;    // [B,K], probabilities before priority updates
  double *probabilities, *cdf;   // [B,N]; cdf unused for eta=1
  double *weights, *totals;      // scratch
  void *scratch;
  size_t scratch_bytes;
  int *status;
};

GpuCoordinateSampler createGpuCoordinateSampler(const Tensor &coordinates,
    const Tensor &targets, int K, const SamplingConfig &config = {}, uint64_t seed = 0);
void freeGpuCoordinateSampler(GpuCoordinateSampler &sampler);

// Rebuilds from current priorities (initially ones). Throws on nonfinite,
// negative priorities or numerically collapsed positive CDF intervals.
void gpuCoordinateSamplingProbabilities(GpuCoordinateSampler &sampler);

// With replacement. Stateless Philox mapping: (step<<32)|(draw<<2)|domain,
// domain 0 for selection; 1/2 reserved for later noise/time. step <= UINT32_MAX,
// B*K < 2^30. Same inputs/seed/step replay on the same toolchain/device.
// eta=1 uses rejection-based bounded integers, with no CDF construction.
// Optional borrowed uniforms [B,K] in [0,1) isolate adaptive search in tests;
// unsupported for eta=1. No per-step allocation or point-array CPU transfers.
void gpuSampleCoordinateIndices(GpuCoordinateSampler &sampler, uint64_t step,
                                const double *uniforms = nullptr);
