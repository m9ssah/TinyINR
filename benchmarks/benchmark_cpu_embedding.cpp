#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

void cpu_coordinate_embedding(const float* input, float* output,
                              int B, int N, int D, int F);


static std::vector<float> make_grid(int H, int W)
{
    std::vector<float> input(H * W * 2);

    for (int row = 0; row < H; ++row) {
        for (int col = 0; col < W; ++col) {
            int n = row * W + col;
            input[n * 2 + 0] = W == 1 ? 0.0f : 2.0f * col / float(W - 1) - 1.0f;
            input[n * 2 + 1] = H == 1 ? 0.0f : 2.0f * row / float(H - 1) - 1.0f;
        }
    }

    return input;
}

static double benchmark_cpu(int H, int W, int F)
{
    const int B = 1;
    const int N = H * W;
    const int D = 2;

    std::vector<float> input = make_grid(H, W);
    std::vector<float> output(B * N * D * F * 2);

    for (int i = 0; i < 5; ++i) {
        cpu_coordinate_embedding(input.data(), output.data(), B, N, D, F);
    }

    const int iterations = 50;
    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < iterations; ++i) {
        cpu_coordinate_embedding(input.data(), output.data(), B, N, D, F);
    }

    auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count() /
           iterations;
}

int main()
{
    std::vector<int> sizes = {32, 128, 256, 512};
    std::vector<int> frequencies = {4, 8, 16};

    std::cout << "image_size,num_coordinates,frequency_bands,cpu_ms\n";

    for (int size : sizes) {
        for (int F : frequencies) {
            double cpu_ms = benchmark_cpu(size, size, F);
            std::cout << size << "x" << size << ","
                      << size * size << ","
                      << F << ","
                      << std::fixed << std::setprecision(6)
                      << cpu_ms << "\n";
        }
    }

    return 0;
}