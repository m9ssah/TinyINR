#include "kernel/coordinate_gather.cuh"
#include "kernel/sampling_utils.cuh"

#include <cmath>

namespace {
__global__ void validateIndices(const int64_t *indices, int draws, int N,
                                int *status) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < draws && (indices[i] < 0 || indices[i] >= N)) atomicExch(status, 1);
}

__global__ void gatherAndEmbed(const float *coordinates, const float *targets,
    const int64_t *indices, int draws, int N, int K, int D, int C, int F,
    int E, float *features, float *sampled_targets, int *value_status) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= draws) return;
  const int original = (row / K) * N + static_cast<int>(indices[row]);
  for (int d = 0; d < D; ++d) {
    const float x = coordinates[original * D + d];
    features[row * E + d] = x;
    if (!isfinite(x)) atomicExch(value_status, 1);
    for (int f = 0; f < F; ++f) {
      // Match the CPU's double pi multiplication followed by float rounding.
      const float frequency = static_cast<float>(ldexp(3.14159265358979323846, f));
      const float phase = frequency * x;
      const int offset = row * E + D + 2 * (f * D + d);
      const float sine = sinf(phase), cosine = cosf(phase);
      features[offset] = sine;
      features[offset + 1] = cosine;
      if (!isfinite(phase) || !isfinite(sine) || !isfinite(cosine))
        atomicExch(value_status, 1);
    }
  }
  for (int c = 0; c < C; ++c) {
    const float value = targets[original * C + c];
    sampled_targets[row * C + c] = value;
    if (!isfinite(value)) atomicExch(value_status, 1);
  }
}
} // namespace

void gpuGatherCoordinateBatch(const float *coordinates, const float *targets,
    const int64_t *indices, int B, int N, int K, int D, int C, int F,
    float *features, float *sampled_targets, int *status) {
  using namespace gpu_sampling;
  require(K > 0 && K <= N && F > 0 && F < 128, "Invalid gather K or F");
  const int points = count(B, N), draws = count(B, K);
  count(points, D); count(points, C);
  const int E = count(D, 1 + 2 * F);
  count(draws, E); count(draws, C);
  const void *pointers[] = {coordinates, targets, indices, features, sampled_targets, status};
  for (const void *p : pointers) devicePointer(p);
  resetStatus(status);
  validateIndices<<<compute_grid_size(draws), THREADS_PER_BLOCK>>>(indices, draws, N, status);
  CUDA_CHECK_LAST_ERROR();
  // A host status check is a grid-wide barrier; gather never reads invalid indices.
  checkStatus(status);
  // Index validation is complete; the gather only writes the status flag.
  gatherAndEmbed<<<compute_grid_size(draws), THREADS_PER_BLOCK>>>(coordinates,
      targets, indices, draws, N, K, D, C, F, E, features, sampled_targets, status);
  CUDA_CHECK_LAST_ERROR();
  checkStatus(status);
}
