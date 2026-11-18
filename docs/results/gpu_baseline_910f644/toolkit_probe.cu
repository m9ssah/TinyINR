#include <cub/cub.cuh>
#include <curand_kernel.h>
#include <curand.h>
#include <cstdio>
#define CHECK(call) do { cudaError_t e=(call); if(e!=cudaSuccess) { std::fprintf(stderr,"%s\n",cudaGetErrorString(e)); return 1; } } while(0)
__global__ void random_probe(float* output) {
    curandStatePhilox4_32_10_t state;
    curand_init(7, 0, 0, &state);
    *output=curand_uniform(&state);
}
int main() {
    double *input, *output; float* random;
    CHECK(cudaMalloc(&input,3*sizeof(double)));
    CHECK(cudaMalloc(&output,3*sizeof(double)));
    CHECK(cudaMalloc(&random,sizeof(float)));
    double values[3]={1,2,3}, result[3];
    CHECK(cudaMemcpy(input,values,sizeof(values),cudaMemcpyHostToDevice));
    size_t bytes=0; void* scratch=nullptr;
    CHECK(cub::DeviceScan::InclusiveSum(scratch,bytes,input,output,3));
    CHECK(cudaMalloc(&scratch,bytes));
    CHECK(cub::DeviceScan::InclusiveSum(scratch,bytes,input,output,3));
    random_probe<<<1,1>>>(random);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(result,output,sizeof(result),cudaMemcpyDeviceToHost));
    float draw; CHECK(cudaMemcpy(&draw,random,sizeof(float),cudaMemcpyDeviceToHost));
    if(result[0]!=1 || result[1]!=3 || result[2]!=6 || !(draw>0 && draw<=1)) return 2;
    std::printf("CUB_VERSION=%d CURAND_VERSION=%d scan=1,3,6 Philox=%g PASS\n",CUB_VERSION,CURAND_VERSION,draw);
    CHECK(cudaFree(scratch)); CHECK(cudaFree(random)); CHECK(cudaFree(output)); CHECK(cudaFree(input));
}
