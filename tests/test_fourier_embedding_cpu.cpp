#include <cmath>
#include <iostream>
#include <vector>
#include <cassert>

#include "../src/kernels/coordinate_embedding.cuh"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static float expected(float x, int f, int trig)
{
    float angle = std::pow(2.0f, static_cast<float>(f)) *
                  static_cast<float>(M_PI) * x;
    return trig == 0 ? std::sin(angle) : std::cos(angle);
}

static void run_case(int B, int N, int D, int F)
{
    std::vector<float> input(B * N * D);
    for (int i = 0; i < static_cast<int>(input.size()); ++i) {
        input[i] = -1.0f + 2.0f * i / static_cast<float>(input.size() - 1);
    }

    std::vector<float> output(B * N * D * F * 2, 0.0f);
    cpu_coordinate_embedding(input.data(), output.data(), B, N, D, F);

    for (int idx = 0; idx < B * N * D; ++idx) {
        for (int f = 0; f < F; ++f) {
            for (int trig = 0; trig < 2; ++trig) {
                int out_idx = idx * F * 2 + f * 2 + trig;
                float diff = std::fabs(output[out_idx] -
                                       expected(input[idx], f, trig));
                assert(diff < 1e-5f);
            }
        }
    }
}

int main()
{
    run_case(1, 2, 2, 1);
    run_case(1, 2, 2, 2);
    run_case(1, 4, 3, 4);

    std::cout << "test_fourier_embedding_cpu PASS\n";
    return 0;
}

