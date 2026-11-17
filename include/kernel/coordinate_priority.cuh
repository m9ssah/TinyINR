#pragma once

#include "kernel/coordinate_sampler.cuh"

// Owns buffers; do not copy or mutate dimensions. Reuse on the creating device.
struct GpuCoordinatePriority {
  int B, N, K, C;
  float *residuals;              // [B,K,1], mean squared error across C channels
  float *candidate_priorities;  // [B,N,1], publication only after full validation
  uint64_t *keys, *sorted_keys;  // (global original point * B*K) + draw order
  void *scratch;
  size_t scratch_bytes;
  int *status;
};

GpuCoordinatePriority createGpuCoordinatePriority(const GpuCoordinateSampler &sampler);
void freeGpuCoordinatePriority(GpuCoordinatePriority &context);

// Borrowed contiguous [B,K,C] inputs. Feed pre-SGD predictions and actual RGB or
// velocity loss targets from gpuTrainStepDevice. Double accumulation, float result.
// Throws on invalid/nonfinite/overflowed values; discard residuals on failure.
void gpuCoordinateResiduals(GpuCoordinatePriority &context,
                            const float *prediction, const float *loss_target);

// Validates all observations, sorts by original point then draw order, averages
// duplicates and applies EMA exactly once per observed point. Unobserved points
// are unchanged. Failed updates never publish candidate priorities. eta=1 is a
// no-op after metadata/beta validation. All storage is persistent; default stream.
void gpuUpdateCoordinatePriorities(GpuCoordinateSampler &sampler,
                                  GpuCoordinatePriority &context, float beta = 0.90f);
