#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

__global__ void coordinate_embedding(const float* d_input, float* d_output, int B, int N, int D, int F)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * N * D;

    if (idx < total)
    {
        float x = d_input[idx];
        for (int f = 0; f < F; f++) 
        {
            float freq = powf(2.0f, f) * M_PI;
            d_output[idx * F * 2 + f * 2] = sinf(freq * x);
            d_output[idx * F * 2 + f * 2 + 1] = cosf(freq * x);
        }
    }
}

int main()
{
    // TEST PURPOSES
    int B = 8;      // batch size
    int N = 1024;   // num points
    int D = 2;      // coordinate dim
    int F = 16;     // frequency bands
    size_t input_size = B * N * D * sizeof(float);
    size_t output_size = B * N * D * 2 * F * sizeof(float);

    // allocate host mem
    float* h_input = (float*)malloc(input_size);
    float* h_output = (float*)malloc(output_size);
    float* h_cpu_check = (float*)malloc(output_size);

    // initialize host mem
    for (int i = 0; i < B * N * D; i++)
        h_input[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;

    // allocate device mem
    float *d_input;
    float *d_output;
    cudaError_t err;

    err = cudaMalloc((void**)&d_input, input_size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device memory for input: %s\n", cudaGetErrorString(err));
        return 1;
    }

    err = cudaMalloc((void**)&d_output, output_size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device memory for output: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // copy host to device
    cudaMemcpy(d_input, h_input, input_size, cudaMemcpyHostToDevice);

    // launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (B * N * D + threadsPerBlock - 1) / threadsPerBlock;

    coordinate_embedding<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, B, N, D, F);
    
    // check for launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch kernel: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // sync
    cudaDeviceSynchronize();

    // copy device to host
    cudaMemcpy(h_output, d_output, output_size, cudaMemcpyDeviceToHost);

    // verify result
    int total_elements = B * N * D;
    for (int idx = 0; idx < total_elements; idx++)
    {
        float x = h_input[idx];
        for (int f = 0; f < F; f++)
        {
            float freq = powf(2.0f, f) * M_PI;
            h_cpu_check[idx * F * 2 + f * 2 + 0] = sinf(freq * x);
            h_cpu_check[idx * F * 2 + f * 2 + 1] = cosf(freq * x);
        }
    }

    bool success = true;
    int total = B * N * D * F * 2;
    for (int i = 0; i < total; i++)
    {
        double diff = abs(h_output[i] - h_cpu_check[i]);
        if (diff > 1e-5)
        {
            fprintf(stderr, "CPU results do not match GPU's at index %d: CPU=%f, GPU=%f\n", 
                i, h_cpu_check[i], h_output[i]);
            success = false;
            break;
        }
    }

    if (success)
        printf("Verification passed\n");

    // free mem
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);
    free(h_cpu_check);

    return 0;
}