#include "kernel/coordinate_priority.cuh"
#include "kernel/sampling_utils.cuh"

#include <cmath>
#include <cfloat>
#include <cub/cub.cuh>

namespace {
__global__ void residualsKernel(const float *prediction, const float *target,
                                float *residuals, int draws, int C, int *status) {
  const int row=blockIdx.x*blockDim.x+threadIdx.x;
  if (row>=draws) return;
  double sum=0;
  for (int c=0; c<C; ++c) {
    const double difference=double(prediction[row*C+c])-double(target[row*C+c]);
    sum+=difference*difference;
  }
  const double mean=sum/C;
  if (!isfinite(mean) || mean>FLT_MAX) { atomicExch(status,1); return; }
  residuals[row]=static_cast<float>(mean);
}

__global__ void validatePriorities(const float *priorities, int points, int *status) {
  const int i=blockIdx.x*blockDim.x+threadIdx.x;
  if (i<points && (!isfinite(priorities[i]) || priorities[i]<0)) atomicExch(status,1);
}

__global__ void observationKeys(const int64_t *indices, const double *probabilities,
    const float *residuals, uint64_t *keys, int draws, int N, int K, int *status) {
  const int i=blockIdx.x*blockDim.x+threadIdx.x;
  if (i>=draws) return;
  if (indices[i]<0 || indices[i]>=N || !isfinite(probabilities[i]) ||
      probabilities[i]<=0 || probabilities[i]>1 || !isfinite(residuals[i]) || residuals[i]<0) {
    atomicExch(status,1); return;
  }
  const uint64_t original=uint64_t(i/K)*N+uint64_t(indices[i]);
  keys[i]=original*uint64_t(draws)+i;
}

__global__ void groupedUpdate(const uint64_t *keys, const float *residuals,
    const float *priorities, float *candidates, int draws, double beta, int *status) {
  const int i=blockIdx.x*blockDim.x+threadIdx.x;
  if (i>=draws) return;
  const uint64_t original=keys[i]/draws;
  if (i>0 && keys[i-1]/draws==original) return;
  double sum=0;
  int end=i;
  while (end<draws && keys[end]/draws==original) {
    sum+=double(residuals[keys[end]%draws]);
    ++end;
  }
  const double mean=sum/(end-i);
  const double updated=beta*double(priorities[original])+(1-beta)*mean;
  if (!isfinite(updated) || updated<0 || updated>FLT_MAX) { atomicExch(status,1); return; }
  candidates[original]=static_cast<float>(updated);
}
} // namespace

GpuCoordinatePriority createGpuCoordinatePriority(const GpuCoordinateSampler &s) {
  using namespace gpu_sampling;
  const int points=count(s.B,s.N), draws=count(s.B,s.K);
  require(s.K<=s.N, "Invalid priority sample count");
  count(draws,s.C);
  require(uint64_t(points)<=UINT64_MAX/uint64_t(draws), "Priority sort key overflow");
  devicePointer(s.priorities);
  GpuCoordinatePriority p{};
  p.B=s.B; p.N=s.N; p.K=s.K; p.C=s.C;
  p.residuals=allocate<float>(draws); p.candidate_priorities=allocate<float>(points);
  p.keys=allocate<uint64_t>(draws); p.sorted_keys=allocate<uint64_t>(draws);
  p.status=allocate<int>(1);
  CUDA_CHECK(cub::DeviceRadixSort::SortKeys(nullptr,p.scratch_bytes,p.keys,p.sorted_keys,draws));
  if (p.scratch_bytes==0) p.scratch_bytes=1;
  p.scratch=allocate<unsigned char>(p.scratch_bytes);
  return p;
}

void freeGpuCoordinatePriority(GpuCoordinatePriority &p) {
  cudaFree(p.residuals); cudaFree(p.candidate_priorities); cudaFree(p.keys);
  cudaFree(p.sorted_keys); cudaFree(p.scratch); cudaFree(p.status);
  p=GpuCoordinatePriority{};
}

void gpuCoordinateResiduals(GpuCoordinatePriority &p, const float *prediction,
                            const float *loss_target) {
  using namespace gpu_sampling;
  const int draws=count(p.B,p.K);
  count(draws,p.C);
  devicePointer(prediction); devicePointer(loss_target); devicePointer(p.status);
  resetStatus(p.status);
  residualsKernel<<<compute_grid_size(draws),THREADS_PER_BLOCK>>>(prediction,
      loss_target,p.residuals,draws,p.C,p.status);
  CUDA_CHECK_LAST_ERROR();
  checkStatus(p.status);
}

void gpuUpdateCoordinatePriorities(GpuCoordinateSampler &s, GpuCoordinatePriority &p,
                                  float beta) {
  using namespace gpu_sampling;
  require(s.B==p.B && s.N==p.N && s.K==p.K && s.C==p.C,
          "Priority context does not match sampler dimensions");
  require(std::isfinite(beta) && beta>=0 && beta<1, "EMA beta must be in [0,1)");
  require(std::isfinite(s.config.eta) && s.config.eta>0 && s.config.eta<=1,
          "Invalid exploration fraction");
  devicePointer(s.priorities); devicePointer(p.status);
  if (s.config.eta==1) return;
  const int points=count(s.B,s.N), draws=count(s.B,s.K);
  resetStatus(p.status);
  validatePriorities<<<compute_grid_size(points),THREADS_PER_BLOCK>>>(s.priorities,points,p.status);
  CUDA_CHECK_LAST_ERROR();
  observationKeys<<<compute_grid_size(draws),THREADS_PER_BLOCK>>>(s.indices,
      s.draw_probabilities,p.residuals,p.keys,draws,s.N,s.K,p.status);
  CUDA_CHECK_LAST_ERROR();
  checkStatus(p.status);
  size_t bytes=p.scratch_bytes;
  CUDA_CHECK(cub::DeviceRadixSort::SortKeys(p.scratch,bytes,p.keys,p.sorted_keys,draws));
  CUDA_CHECK(cudaMemcpyAsync(p.candidate_priorities,s.priorities,points*sizeof(float),cudaMemcpyDeviceToDevice));
  groupedUpdate<<<compute_grid_size(draws),THREADS_PER_BLOCK>>>(p.sorted_keys,
      p.residuals,s.priorities,p.candidate_priorities,draws,beta,p.status);
  CUDA_CHECK_LAST_ERROR();
  checkStatus(p.status);
  CUDA_CHECK(cudaMemcpyAsync(s.priorities,p.candidate_priorities,points*sizeof(float),cudaMemcpyDeviceToDevice));
}
