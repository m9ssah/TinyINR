#include <iostream>
#include <limits>

#include "gpu_sampling_test_utils.cuh"
#include "kernel/coordinate_gather.cuh"
#include "ops/coordinate_batch.h"
#include "ops/fourier_embedding.h"

using namespace sampling_test;

void parity(int B, int N, int K, int D, int C, int F) {
  Tensor coords({B,N,D}), values({B,N,C});
  for (int64_t i=0; i<coords.numel(); ++i) coords.data()[i] = (i%19-9)*0.03125f;
  for (int64_t i=0; i<values.numel(); ++i) values.data()[i] = float(i)*0.125f;
  std::vector<int64_t> indices(B*K);
  for (int i=0; i<B*K; ++i) indices[i] = (i%3 == 0) ? N-1 : (i%N);
  auto gathered = gatherCoordinateBatch(CoordinateBatch(coords, values), indices, K);
  auto reference = fourierEmbedding(gathered.coordinates(), F);
  Buffer<float> dc(std::vector<float>(coords.data(),coords.data()+coords.numel()));
  Buffer<float> dv(std::vector<float>(values.data(),values.data()+values.numel()));
  Buffer<int64_t> di(indices);
  Buffer<float> output(reference.numel()), targets(B*K*C);
  Buffer<int> status(1);
  auto run = [&] { gpuGatherCoordinateBatch(dc.data,dv.data,di.data,B,N,K,D,C,F,
                                           output.data,targets.data,status.data); };
  run();
  auto actual=output.read(), rgb=targets.read();
  for (size_t i=0; i<actual.size(); ++i) near(actual[i],reference.data()[i],1e-5,1e-4);
  for (int i=0; i<B*K; ++i)
    for (int d=0; d<D; ++d)
      check(actual[i*(D+2*D*F)+d] == gathered.coordinates().data()[i*D+d], "Raw coordinates changed");
  for (size_t i=0; i<rgb.size(); ++i) check(rgb[i]==gathered.values().data()[i],"RGB mismatch");
  indices.back()=N; di.upload(indices);
  rejects(run);
  check(output.read()==actual,"Invalid index changed output before rejection");
  indices.back()=-1; di.upload(indices); rejects(run);
  indices.back()=0; di.upload(indices);
  auto bad=dc.read(); bad[0]=std::numeric_limits<float>::quiet_NaN(); dc.upload(bad);
  indices[0]=0; di.upload(indices); rejects(run);
  bad[0]=std::numeric_limits<float>::max(); dc.upload(bad); rejects(run);
}

int main() {
  try {
    parity(2,7,7,2,3,8);
    parity(2,9,1,3,2,16);
    parity(2,263,259,2,3,16);
    parity(1,1,1,1,1,1);
    rejects([] { gpuGatherCoordinateBatch(nullptr,nullptr,nullptr,1,1,2,2,3,8,nullptr,nullptr,nullptr); });
    rejects([] { gpuGatherCoordinateBatch(nullptr,nullptr,nullptr,1,1,1,2,3,128,nullptr,nullptr,nullptr); });
    rejects([] { gpuGatherCoordinateBatch(nullptr,nullptr,nullptr,2147483647,2,1,2,3,8,nullptr,nullptr,nullptr); });
    std::cout << "test_coordinate_gather_cuda PASS\n";
  } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
