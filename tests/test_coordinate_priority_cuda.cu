#include <iostream>
#include <limits>

#include "gpu_sampling_test_utils.cuh"
#include "kernel/coordinate_priority.cuh"
#include "training/cicfm.h"

using namespace sampling_test;

std::vector<float> download(const float *data, int count) {
  std::vector<float> values(count);
  cuda_d2h(values.data(),data,count);
  return values;
}

void updates(int N, int K, bool all_duplicate, float beta) {
  Tensor coords({2,N,2}), targets({2,N,3}), original({2,N,1});
  auto s=createGpuCoordinateSampler(coords,targets,K);
  auto p=createGpuCoordinatePriority(s);
  SampleSelection selection{2,N,K,std::vector<int64_t>(2*K),std::vector<double>(2*K,1.0/N)};
  Tensor residuals({2,K,1});
  for (int i=0; i<2*N; ++i) original.data()[i]=float(i%11);
  for (int i=0; i<2*K; ++i) {
    selection.indices[i]=all_duplicate ? 0 : (i%3==0 ? N-1 : i%N);
    residuals.data()[i]=float(i%17)*0.25f;
  }
  cuda_h2d(s.priorities,original.data(),2*N);
  CUDA_CHECK(cudaMemcpy(s.indices,selection.indices.data(),2*K*sizeof(int64_t),cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(s.draw_probabilities,selection.probabilities.data(),2*K*sizeof(double),cudaMemcpyHostToDevice));
  cuda_h2d(p.residuals,residuals.data(),2*K);
  Tensor reference=original;
  updateCoordinatePriorities(reference,selection,residuals,beta);
  gpuUpdateCoordinatePriorities(s,p,beta);
  auto first=download(s.priorities,2*N);
  for (int i=0; i<2*N; ++i) {
    near(first[i],reference.data()[i]);
    bool observed=false;
    for (int k=0; k<K; ++k)
      observed=observed || selection.indices[(i/N)*K+k]==i%N;
    if (!observed) check(first[i]==original.data()[i],"Unobserved priority changed");
  }
  cuda_h2d(s.priorities,original.data(),2*N);
  gpuUpdateCoordinatePriorities(s,p,beta);
  check(download(s.priorities,2*N)==first,"Duplicate update did not replay exactly");
  cuda_h2d(s.priorities,original.data(),2*N);
  const auto unchanged=download(s.priorities,2*N);
  auto fail=[&] {
    rejects([&] { gpuUpdateCoordinatePriorities(s,p,beta); });
    check(download(s.priorities,2*N)==unchanged,"Failed update partially changed priorities");
  };
  residuals.data()[2*K-1]=std::numeric_limits<float>::quiet_NaN();
  cuda_h2d(p.residuals,residuals.data(),2*K); fail();
  residuals.data()[2*K-1]=-1;
  cuda_h2d(p.residuals,residuals.data(),2*K); fail();
  residuals.data()[2*K-1]=1;
  cuda_h2d(p.residuals,residuals.data(),2*K);
  selection.indices.back()=N;
  CUDA_CHECK(cudaMemcpy(s.indices,selection.indices.data(),2*K*sizeof(int64_t),cudaMemcpyHostToDevice));
  fail();
  selection.indices.back()=0;
  CUDA_CHECK(cudaMemcpy(s.indices,selection.indices.data(),2*K*sizeof(int64_t),cudaMemcpyHostToDevice));
  selection.probabilities.back()=0;
  CUDA_CHECK(cudaMemcpy(s.draw_probabilities,selection.probabilities.data(),2*K*sizeof(double),cudaMemcpyHostToDevice));
  fail();
  selection.probabilities.back()=1.0/N;
  CUDA_CHECK(cudaMemcpy(s.draw_probabilities,selection.probabilities.data(),2*K*sizeof(double),cudaMemcpyHostToDevice));
  original.data()[2*N-1]=-1; cuda_h2d(s.priorities,original.data(),2*N);
  const auto invalid_priorities=download(s.priorities,2*N);
  rejects([&] { gpuUpdateCoordinatePriorities(s,p,beta); });
  check(download(s.priorities,2*N)==invalid_priorities,"Invalid priorities partly updated");
  original.data()[2*N-1]=1; cuda_h2d(s.priorities,original.data(),2*N);
  s.config.eta=1;
  const auto uniform_before=download(s.priorities,2*N);
  gpuUpdateCoordinatePriorities(s,p,beta);
  check(download(s.priorities,2*N)==uniform_before,"Uniform mode updated priorities");
  rejects([&] { gpuUpdateCoordinatePriorities(s,p,1); });
  freeGpuCoordinatePriority(p); freeGpuCoordinateSampler(s);
}

void residuals(bool flow) {
  Tensor coords({2,3,2}), rgb({2,3,3}), z0({2,3,3}), t({2,3,1}), features({2,3,2});
  for (int i=0; i<18; ++i) { rgb.data()[i]=i*0.1f; z0.data()[i]=-0.3f+i*0.02f; }
  for (int i=0; i<6; ++i) t.data()[i]=0.3f;
  auto batch=makeCicfmBatch(features,rgb,z0,t);
  const Tensor &target=flow ? batch.target_velocity : rgb;
  Tensor prediction({2,3,3});
  for (int i=0; i<18; ++i) prediction.data()[i]=target.data()[i]+float(i%3-1);
  auto s=createGpuCoordinateSampler(coords,rgb,3);
  auto p=createGpuCoordinatePriority(s);
  Buffer<float> dp(std::vector<float>(prediction.data(),prediction.data()+18));
  Buffer<float> dt(std::vector<float>(target.data(),target.data()+18));
  gpuCoordinateResiduals(p,dp.data,dt.data);
  auto result=download(p.residuals,6);
  for (int i=0; i<6; ++i) {
    double sum=0;
    for (int c=0; c<3; ++c) {
      double d=double(prediction.data()[i*3+c])-target.data()[i*3+c]; sum+=d*d;
    }
    near(result[i],float(sum/3)); near(result[i],2.0/3.0);
  }
  auto bad=dp.read(); bad.back()=std::numeric_limits<float>::infinity(); dp.upload(bad);
  rejects([&] { gpuCoordinateResiduals(p,dp.data,dt.data); });
  bad.back()=std::numeric_limits<float>::max(); dp.upload(bad);
  rejects([&] { gpuCoordinateResiduals(p,dp.data,dt.data); });
  rejects([&] { gpuCoordinateResiduals(p,nullptr,dt.data); });
  freeGpuCoordinatePriority(p); freeGpuCoordinateSampler(s);
}

int main() {
  try {
    updates(7,5,false,0.9f); updates(7,5,false,0); updates(263,263,true,0.9f);
    updates(1,1,true,0.9f); residuals(false); residuals(true);
    std::cout << "test_coordinate_priority_cuda PASS\n";
  } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
