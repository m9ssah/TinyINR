#include "kernel/coordinate_sampler.cuh"
#include "kernel/sampling_utils.cuh"

#include <cmath>
#include <cub/cub.cuh>
#include <curand_kernel.h>

namespace {
void validateConfig(const SamplingConfig &config) {
  gpu_sampling::require(std::isfinite(config.eta) && config.eta > 0 && config.eta <= 1 &&
      std::isfinite(config.epsilon) && config.epsilon > 0, "Invalid sampling configuration");
}

__global__ void initializePriorities(float *priorities, int count) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) priorities[i] = 1;
}

__global__ void prepareWeights(const float *priorities, double *weights,
    double *probabilities, int count, int N, double epsilon, bool uniform, int *status) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const float priority = priorities[i];
  if (!isfinite(priority) || priority < 0) atomicExch(status, 1);
  weights[i] = double(priority) + epsilon;
  if (uniform) probabilities[i] = 1.0 / N;
}

__global__ void probabilitiesKernel(const double *weights, const double *totals,
    double *probabilities, int count, int N, double eta, int *status) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const double total = totals[i/N];
  const double p = eta/N + (1-eta)*(weights[i]/total);
  probabilities[i] = p;
  if (!(total > 0) || !isfinite(total) || !(p > 0) || !isfinite(p)) atomicExch(status, 1);
}

// CUB 3.3 DeviceScan does not guarantee floating-point replay. Fixed-order
// BlockScan tiles preserve it without a custom scan primitive.
__global__ void buildCdf(const double *probabilities, double *cdf, double *totals, int N) {
  using Scan=cub::BlockScan<double,THREADS_PER_BLOCK>;
  __shared__ typename Scan::TempStorage scratch;
  __shared__ double prefix;
  const int b=blockIdx.x, lane=threadIdx.x;
  if (lane==0) prefix=0;
  __syncthreads();
  // shortcut: one block per image; revisit large-N throughput after profiling.
  for (int start=0; start<N; start+=THREADS_PER_BLOCK) {
    const int i=start+lane;
    const double value=i<N ? probabilities[b*N+i] : 0;
    double partial, aggregate;
    Scan(scratch).InclusiveSum(value,partial,aggregate);
    if (i<N) cdf[b*N+i]=prefix+partial;
    __syncthreads();
    if (lane==0) prefix+=aggregate;
    __syncthreads();
  }
  // Normalize by the last stored CDF value, including its actual rounding.
  if (lane==0) totals[b]=cdf[(b+1)*N-1];
}

__global__ void normalize(double *probabilities, double *cdf, const double *totals,
                           int count, int N) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) { probabilities[i] /= totals[i/N]; cdf[i] /= totals[i/N]; }
}

__global__ void validateCdf(const double *probabilities, const double *cdf,
                            int count, int N, int *status) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const double previous = i%N == 0 ? 0 : cdf[i-1];
  if (!isfinite(probabilities[i]) || !(probabilities[i] > 0) || probabilities[i] > 1 ||
      !isfinite(cdf[i]) || !(cdf[i] > previous) || cdf[i] > 1 ||
      (i%N == N-1 && cdf[i] != 1)) atomicExch(status, 1);
}

__global__ void validateUniforms(const double *uniforms, int count, int *status) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count && (!isfinite(uniforms[i]) || uniforms[i] < 0 || uniforms[i] >= 1))
    atomicExch(status, 1);
}

__global__ void draw(const double *cdf, const double *probabilities, int64_t *indices,
    double *draw_probabilities, int draws, int N, int K, uint64_t seed,
    uint64_t step, bool uniform, const double *uniforms, int *status) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= draws) return;
  curandStatePhilox4_32_10_t rng;
  const uint64_t sequence = (step << 32) | (uint64_t(i) << 2);
  curand_init(seed, sequence, 0, &rng);
  int selected;
  if (uniform) {
    const unsigned bound = static_cast<unsigned>(N);
    const unsigned threshold = (0u-bound)%bound;
    unsigned value;
    do { value = curand(&rng); } while (value < threshold);
    selected = static_cast<int>(value%bound);
  } else {
    const double u = uniforms ? uniforms[i] : 1.0-curand_uniform_double(&rng);
    const int base = (i/K)*N;
    int lo=0, hi=N;
    while (lo < hi) {
      const int mid=lo+(hi-lo)/2;
      if (cdf[base+mid] > u) hi=mid; else lo=mid+1;
    }
    if (lo == N) { atomicExch(status, 1); return; }
    selected=lo;
  }
  indices[i]=selected;
  draw_probabilities[i]=uniform ? 1.0/N : probabilities[(i/K)*N+selected];
}
} // namespace

GpuCoordinateSampler createGpuCoordinateSampler(const Tensor &coordinates,
    const Tensor &targets, int K, const SamplingConfig &config, uint64_t seed) {
  using namespace gpu_sampling;
  validateConfig(config);
  require(coordinates.ndim()==3 && targets.ndim()==3 &&
          coordinates.shape()[0]==targets.shape()[0] && coordinates.shape()[1]==targets.shape()[1],
          "Sampler data must have matching [B,N,D] and [B,N,C] shapes");
  const int points=count(coordinates.shape()[0],coordinates.shape()[1]);
  count(points,coordinates.shape()[2]); count(points,targets.shape()[2]);
  require(K > 0 && K <= coordinates.shape()[1], "Sample count must be in [1,N]");
  const int draws=count(coordinates.shape()[0],K);
  require(draws < (1<<30), "Philox draw index exceeds reserved sequence bits");
  for (const Tensor *tensor : {&coordinates,&targets})
    for (int64_t i=0; i<tensor->numel(); ++i)
      require(std::isfinite(tensor->data()[i]), "Sampler source data must be finite");
  GpuCoordinateSampler s{};
  s.B=static_cast<int>(coordinates.shape()[0]); s.N=static_cast<int>(coordinates.shape()[1]);
  s.D=static_cast<int>(coordinates.shape()[2]); s.C=static_cast<int>(targets.shape()[2]);
  s.K=K; s.config=config; s.seed=seed;
  s.coordinates=allocate<float>(coordinates.numel()); s.targets=allocate<float>(targets.numel());
  cuda_h2d(s.coordinates,coordinates.data(),coordinates.numel());
  cuda_h2d(s.targets,targets.data(),targets.numel());
  s.priorities=allocate<float>(points); s.indices=allocate<int64_t>(draws);
  s.draw_probabilities=allocate<double>(draws);
  s.probabilities=allocate<double>(points); s.cdf=allocate<double>(points);
  s.weights=allocate<double>(points); s.totals=allocate<double>(s.B); s.status=allocate<int>(1);
  CUDA_CHECK(cub::DeviceReduce::Sum(nullptr,s.scratch_bytes,s.weights,s.totals,s.N));
  if (s.scratch_bytes==0) s.scratch_bytes=1;
  s.scratch=allocate<unsigned char>(s.scratch_bytes);
  initializePriorities<<<compute_grid_size(points),THREADS_PER_BLOCK>>>(s.priorities,points);
  CUDA_CHECK_LAST_ERROR();
  return s;
}

void freeGpuCoordinateSampler(GpuCoordinateSampler &s) {
  cudaFree(s.coordinates); cudaFree(s.targets); cudaFree(s.priorities);
  cudaFree(s.indices); cudaFree(s.draw_probabilities); cudaFree(s.probabilities);
  cudaFree(s.cdf); cudaFree(s.weights); cudaFree(s.totals); cudaFree(s.scratch); cudaFree(s.status);
  s=GpuCoordinateSampler{};
}

void gpuCoordinateSamplingProbabilities(GpuCoordinateSampler &s) {
  using namespace gpu_sampling;
  validateConfig(s.config); devicePointer(s.priorities); devicePointer(s.status);
  const int points=count(s.B,s.N);
  resetStatus(s.status);
  prepareWeights<<<compute_grid_size(points),THREADS_PER_BLOCK>>>(s.priorities,s.weights,
      s.probabilities,points,s.N,s.config.epsilon,s.config.eta==1,s.status);
  CUDA_CHECK_LAST_ERROR();
  checkStatus(s.status);
  if (s.config.eta==1) return;
  for (int b=0; b<s.B; ++b) {
    size_t bytes=s.scratch_bytes;
    CUDA_CHECK(cub::DeviceReduce::Sum(s.scratch,bytes,s.weights+b*s.N,s.totals+b,s.N));
  }
  probabilitiesKernel<<<compute_grid_size(points),THREADS_PER_BLOCK>>>(s.weights,s.totals,
      s.probabilities,points,s.N,s.config.eta,s.status);
  CUDA_CHECK_LAST_ERROR();
  buildCdf<<<s.B,THREADS_PER_BLOCK>>>(s.probabilities,s.cdf,s.totals,s.N);
  CUDA_CHECK_LAST_ERROR();
  normalize<<<compute_grid_size(points),THREADS_PER_BLOCK>>>(s.probabilities,s.cdf,s.totals,points,s.N);
  CUDA_CHECK_LAST_ERROR();
  validateCdf<<<compute_grid_size(points),THREADS_PER_BLOCK>>>(s.probabilities,s.cdf,points,s.N,s.status);
  CUDA_CHECK_LAST_ERROR();
  checkStatus(s.status);
}

void gpuSampleCoordinateIndices(GpuCoordinateSampler &s, uint64_t step, const double *uniforms) {
  using namespace gpu_sampling;
  require(step <= UINT32_MAX, "Philox step exceeds reserved sequence bits");
  const int draws=count(s.B,s.K);
  require(draws < (1<<30), "Philox draw index exceeds reserved sequence bits");
  if (uniforms) {
    require(s.config.eta != 1, "Injected uniforms require adaptive sampling");
    devicePointer(uniforms);
  }
  gpuCoordinateSamplingProbabilities(s);
  if (uniforms) {
    validateUniforms<<<compute_grid_size(draws),THREADS_PER_BLOCK>>>(uniforms,draws,s.status);
    CUDA_CHECK_LAST_ERROR();
    checkStatus(s.status);
  }
  draw<<<compute_grid_size(draws),THREADS_PER_BLOCK>>>(s.cdf,s.probabilities,s.indices,
      s.draw_probabilities,draws,s.N,s.K,s.seed,step,s.config.eta==1,uniforms,s.status);
  CUDA_CHECK_LAST_ERROR();
  checkStatus(s.status);
}
