#include <cassert>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include "../src/kernels/cuda_utils.cuh"
#include "../src/kernels/coordinate_embedding.cuh"

struct Case {
    int B;
    int N;
    int D;
    int F;
};

static float max_abs_error(const std::vector<float>& a,
                           const std::vector<float>& b)
{
    float max_err = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        max_err = std::max(max_err, fabsf(a[i] - b[i]));
    }
    return max_err;
}

static void run_case(const Case& c)
{
    int input_count = c.B * c.N * c.D;
    int output_count = c.B * c.N * c.D * c.F * 2;

    // allocating host memory 
    std::vector<float> input(input_count);
    std::vector<float> cpu_output(output_count);
    std::vector<float> gpu_output(output_count);

    // deterministic random number generator (so every run gets the same random inputs)
    std::mt19937 rng(123);

    // creates floats between -1.0 and 1.0 (matches the normalized coordinate range)
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (float& x : input) {
        x = dist(rng);
    }

    // running cpu version of embedding 
    cpu_coordinate_embedding(input.data(), cpu_output.data(),
                             c.B, c.N, c.D, c.F);

    
    // gpu memory allocation and kernel setup 
    float* d_input = cuda_alloc(input_count);
    float* d_output = cuda_alloc(output_count);

    cuda_h2d(d_input, input.data(), input_count);

    int threads = THREADS_PER_BLOCK;
    int blocks = compute_grid_size(input_count, threads);

    coordinate_embedding<<<blocks, threads>>>(d_input, d_output,
                                              c.B, c.N, c.D, c.F);
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK(cudaDeviceSynchronize());

    cuda_d2h(gpu_output.data(), d_output, output_count);

    // comparing cpu and gpu outputs 
    float err = max_abs_error(cpu_output, gpu_output);
    std::cout << "B=" << c.B << " N=" << c.N << " D=" << c.D
              << " F=" << c.F << " max_abs_error=" << err << "\n";

    assert(err < 1e-5f);

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
}

int main()
{
    std::vector<Case> cases = {
        {1, 1024, 2, 4},
        {1, 1024, 2, 8},
        {1, 1024, 2, 16},
        {1, 16384, 2, 4},
        {1, 16384, 2, 8},
        {1, 16384, 2, 16},
        {1, 65536, 2, 8},
        {1, 256, 3, 8}
    };

    for (const Case& c : cases) {
        run_case(c);
    }

    std::cout << "test_cpu_cuda_parity PASS\n";
    return 0;
}