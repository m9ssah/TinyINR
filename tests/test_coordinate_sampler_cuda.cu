#include <algorithm>
#include <iostream>
#include <limits>

#include "gpu_sampling_test_utils.cuh"
#include "kernel/coordinate_sampler.cuh"

using namespace sampling_test;

template <typename T> std::vector<T> read(const T *data, int n) {
  std::vector<T> result(n);
  CUDA_CHECK(cudaMemcpy(result.data(),data,n*sizeof(T),cudaMemcpyDeviceToHost));
  return result;
}

void probabilitiesAndBoundaries() {
  Tensor coords({2,7,2}), targets({2,7,3}), priorities({2,7,1});
  auto s=createGpuCoordinateSampler(coords,targets,7,{},17);
  for (int i=0; i<14; ++i) priorities.data()[i]=i<7 ? 0 : float(i-7);
  cuda_h2d(s.priorities,priorities.data(),14);
  gpuCoordinateSamplingProbabilities(s);
  auto expected=coordinateSamplingProbabilities(priorities,s.config);
  auto p=read(s.probabilities,14), cdf=read(s.cdf,14);
  for (int b=0; b<2; ++b) {
    double sum=0;
    for (int i=0; i<7; ++i) {
      const int j=b*7+i;
      near(p[j],expected[j],1e-12,1e-10);
      check(p[j]>=double(s.config.eta)/7-1e-14,"Exploration floor lost");
      sum+=p[j]; near(cdf[j],sum,1e-12,1e-10);
    }
    near(sum,1,1e-12,0);
  }
  std::vector<double> u(14,0);
  for (int b=0; b<2; ++b) {
    u[b*7]=0; u[b*7+1]=cdf[b*7];
    u[b*7+2]=std::nextafter(cdf[b*7],0.0);
    u[b*7+3]=std::nextafter(1.0,0.0);
    u[b*7+4]=cdf[b*7+5];
  }
  Buffer<double> uniforms(u);
  gpuSampleCoordinateIndices(s,0,uniforms.data);
  auto indices=read(s.indices,14);
  auto selected=read(s.draw_probabilities,14);
  for (int b=0; b<2; ++b) {
    check(indices[b*7]==0 && indices[b*7+1]==1 && indices[b*7+2]==0 &&
          indices[b*7+3]==6 && indices[b*7+4]==6,"CDF boundary search failed");
    for (int i=0; i<7; ++i) near(selected[b*7+i],p[b*7+indices[b*7+i]],0,0);
  }
  u.back()=1; uniforms.upload(u);
  rejects([&] { gpuSampleCoordinateIndices(s,0,uniforms.data); });
  check(read(s.indices,14)==indices,"Invalid uniforms changed selection");
  priorities.data()[13]=-1; cuda_h2d(s.priorities,priorities.data(),14);
  rejects([&] { gpuSampleCoordinateIndices(s,0); });
  check(read(s.indices,14)==indices,"Invalid later batch changed selection");
  s.config.eta=1;
  rejects([&] { gpuSampleCoordinateIndices(s,0); });
  priorities.data()[13]=std::numeric_limits<float>::infinity();
  cuda_h2d(s.priorities,priorities.data(),14);
  rejects([&] { gpuSampleCoordinateIndices(s,0); });
  freeGpuCoordinateSampler(s);
}

void replayAndFrequency(bool uniform) {
  const int N=257;
  Tensor coords({1,N,2}), targets({1,N,3}), priorities({1,N,1});
  SamplingConfig config; config.eta=uniform ? 1 : 0.1f;
  auto s=createGpuCoordinateSampler(coords,targets,N,config,1234);
  for (int i=0; i<N; ++i) priorities.data()[i]=i==7 ? 100 : 1;
  cuda_h2d(s.priorities,priorities.data(),N);
  gpuSampleCoordinateIndices(s,0);
  auto first=read(s.indices,N);
  auto prob=read(s.draw_probabilities,N);
  gpuSampleCoordinateIndices(s,0);
  check(read(s.indices,N)==first && read(s.draw_probabilities,N)==prob,"Seed replay failed");
  gpuSampleCoordinateIndices(s,1);
  check(read(s.indices,N)!=first,"Step did not advance random sequence");
  std::vector<int> histogram(N,0);
  for (int step=0; step<128; ++step) {
    gpuSampleCoordinateIndices(s,step);
    auto indices=read(s.indices,N);
    auto probabilities=read(s.draw_probabilities,N);
    for (int i=0; i<N; ++i) {
      check(indices[i]>=0 && indices[i]<N,"Invalid draw");
      ++histogram[indices[i]];
      if (uniform) near(probabilities[i],1.0/N,0,0);
    }
  }
  auto expected=coordinateSamplingProbabilities(priorities,config);
  for (int i=0; i<N; ++i) {
    const double mean=128*N*expected[i];
    check(std::fabs(histogram[i]-mean)<=6*std::sqrt(mean*(1-expected[i]))+8,
          "Draw frequencies outside predeclared six-sigma allowance");
  }
  rejects([&] { gpuSampleCoordinateIndices(s,uint64_t(UINT32_MAX)+1); });
  freeGpuCoordinateSampler(s);
}

void multiTileReplay() {
  const int B=2, N=4099, K=513;
  Tensor coords({B,N,2}), targets({B,N,3}), priorities({B,N,1});
  auto s=createGpuCoordinateSampler(coords,targets,K,{},4321);
  for (int i=0; i<B*N; ++i) priorities.data()[i]=float((i*7)%101)*0.001f;
  cuda_h2d(s.priorities,priorities.data(),B*N);
  gpuSampleCoordinateIndices(s,19);
  const auto cdf=read(s.cdf,B*N);
  const auto indices=read(s.indices,B*K);
  const auto actual=read(s.probabilities,B*N);
  const auto expected=coordinateSamplingProbabilities(priorities,s.config);
  for (int i=0; i<B*N; ++i) near(actual[i],expected[i],1e-12,1e-10);
  for (int replay=0; replay<8; ++replay) {
    gpuSampleCoordinateIndices(s,19);
    check(read(s.cdf,B*N)==cdf && read(s.indices,B*K)==indices,"Multi-tile replay changed");
  }
  freeGpuCoordinateSampler(s);
}

void singletonAndFailures() {
  Tensor coords({2,1,1}), targets({2,1,1});
  auto s=createGpuCoordinateSampler(coords,targets,1);
  gpuSampleCoordinateIndices(s,UINT32_MAX);
  check(read(s.indices,2)==std::vector<int64_t>({0,0}),"Singleton indices");
  check(read(s.draw_probabilities,2)==std::vector<double>({1,1}),"Singleton probabilities");
  s.config.eta=0;
  rejects([&] { gpuSampleCoordinateIndices(s,0); });
  freeGpuCoordinateSampler(s);
  rejects([&] { createGpuCoordinateSampler(coords,targets,2); });
  coords.data()[0]=std::numeric_limits<float>::quiet_NaN();
  rejects([&] { createGpuCoordinateSampler(coords,targets,1); });
  Tensor two({1,2,1});
  auto skew=createGpuCoordinateSampler(two,two,2);
  const float priorities[]={std::numeric_limits<float>::max(),0};
  cuda_h2d(skew.priorities,priorities,2);
  skew.config.eta=std::numeric_limits<float>::denorm_min();
  rejects([&] { gpuSampleCoordinateIndices(skew,0); });
  freeGpuCoordinateSampler(skew);
}

int main() {
  try {
    probabilitiesAndBoundaries(); replayAndFrequency(false); replayAndFrequency(true);
    multiTileReplay(); singletonAndFailures();
    std::cout << "test_coordinate_sampler_cuda PASS\n";
  } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
