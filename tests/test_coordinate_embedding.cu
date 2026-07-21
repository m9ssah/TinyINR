#include <stdio.h>
#include <stdlib.h>

#include "../src/kernels/cuda_utils.cuh"
#include "../src/kernels/coordinate_embedding.cuh"

int main()
{
    int B = 8;
    int N = 1024;
    int D = 2;
    int F = 16;

    int    input_count  = B * N * D;
    int    output_count = B * N * D * F * 2;
    
    size_t input_size  = input_count  * sizeof(float);
    size_t output_size = output_count * sizeof(float);

    // allocate host memory
    float* h_input   = (float*)malloc(input_size);
    float* h_gpu_out = (float*)malloc(output_size);
    float* h_cpu_out = (float*)malloc(output_size);

    // initialize memory
    for (int i = 0; i < input_count; i++)
        h_input[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;

    // allocate device memory
    float* d_input  = cuda_alloc(input_count);
    float* d_output = cuda_alloc(output_count);

    // host to device
    cuda_h2d(d_input, h_input, input_count);

    // launch kernel
    int threads = THREADS_PER_BLOCK;
    int blocks  = compute_grid_size(input_count, threads);
    printf("Launching kernel: %d blocks x %d threads\n", blocks, threads);

    coordinate_embedding<<<blocks, threads>>>(d_input, d_output, B, N, D, F);
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK(cudaDeviceSynchronize());

    // device to host
    cuda_d2h(h_gpu_out, d_output, output_count);

    // cpu parity check
    cpu_coordinate_embedding(h_input, h_cpu_out, B, N, D, F);

    bool test = check_parity(h_cpu_out, h_gpu_out, output_count);
    printf("Test result: %s\n", test ? "PASS" : "FAIL");

    // cleanup
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    free(h_input);
    free(h_gpu_out);
    free(h_cpu_out);

    return test ? 0 : 1;
}
