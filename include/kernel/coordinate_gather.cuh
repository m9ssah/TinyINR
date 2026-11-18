#pragma once

#include <cstdint>

// Default stream; all buffers are borrowed contiguous device allocations, with
// no overlap. Coordinates [B,N,D], targets [B,N,C], local indices [B,K].
// Outputs: features [B,K,D+2*D*F], sampled targets [B,K,C]; status is one int.
// Requires 1<=K<=N, 0<F<128 and adequate allocation capacity from the caller.
// Raw coordinates precede frequency-major sin/cos pairs, matching CPU training.
// No allocations/uploads. Throws on invalid metadata/data; discard outputs on
// failure. Invalid indices gate the gather globally before any indexed read.
void gpuGatherCoordinateBatch(const float *coordinates, const float *targets,
    const int64_t *indices, int B, int N, int K, int D, int C, int F,
    float *features, float *sampled_targets, int *status);
