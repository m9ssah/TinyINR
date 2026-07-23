#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#include "../src/kernels/coordinate_embedding.cuh"
#include "bench_utils.cuh"

// benchmark config
static const int B = 1;
static const int D = 2;
static const int NUM_TRIALS = 20;

// sweep params
static const int N_VALUES[] = {1024, 16384, 65536, 262144};
static const int F_VALUES[] = {4, 8, 16};

static const int count_N = sizeof(N_VALUES) / sizeof(N_VALUES[0]);
static const int count_F = sizeof(F_VALUES) / sizeof(F_VALUES[0]);

int main() {
  printf("Coordinate Embedding Benchmark\n");
  printf("B=%d  D=%d  Trials=%d\n\n", B, D, NUM_TRIALS);

  cuda_warmup();

  L2Flusher flusher;

  CsvWriter csv("benchmarks/results/coordinate_embedding.csv");

  int max_N = N_VALUES[count_N - 1];
  int max_F = F_VALUES[count_F - 1];
  int max_input_count = B * max_N * D;
  int max_output_count = B * max_N * D * max_F * 2;

  // alloc host buffers
  std::vector<float> h_input(max_input_count);
  std::vector<float> h_gpu_out(max_output_count);
  std::vector<float> h_cpu_out(max_output_count);

  // alloc device buffers
  float *d_input = cuda_alloc(max_input_count);
  float *d_output = cuda_alloc(max_output_count);

  GpuTimer gpu_timer;
  CpuTimer cpu_timer;

  // sweep over all combos
  for (int n = 0; < count_N; n++) {
    for (int f = 0; f < count_F; f++) {
      int N = N_VALUES[n];
      int F = F_VALUES[f];

      int input_count = B * N * D;
      int output_count = B * N * D * F * 2;

      int threads = THREADS_PER_BLOCK;
      int blocks = compute_grid_size(input_count, threads);

      // init input
      srand(42);
      for (int i = 0; i < input_count; i++)
        h_input[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;

      std::vector<float> cpu_times(NUM_TRIALS);
      std::vector<float> h2d_times(NUM_TRIALS);
      std::vector<float> kernel_times(NUM_TRIALS);
      std::vector<float> d2h_times(NUM_TRIALS);

      printf("N=%6d  F=%2d  (%d blocks x %d threads)\n", N, F, blocks, threads);

      for (int trial = 0; trial < NUM_TRIALS; trial++) {
        flusher.flush();

        // time h2d
        gpu_timer.start();
        cuda_h2d(d_input, h_input.data(), input_count);
        gpu_timer.stop();
        h2d_times[trial] = gpu_timer.elapsed_ms();

        // time kernel
        gpu_timer.start();
        coordinate_embedding<<<blocks, threads>>>(d_input, d_output, B, N, D,
                                                  F);
        gpu_timer.stop();
        kernel_times[trial] = gpu_timer.elapsed_ms();

        // time d2h
        gpu_timer.start();
        cuda_d2h(h_gpu_out.data(), d_output, output_count);
        gpu_timer.stop();
        d2h_times[trial] = gpu_timer.elapsed_ms();

        // time cpu
        cpu_timer.start();
        cpu_coordinate_embedding(h_input.data(), h_cpu_out.data(), B, N, D, F);
        cpu_timer.stop();
        cpu_times[trial] = cpu_timer.elapsed_ms();
      }

      float med_cpu = median(cpu_times);
      float med_h2d = median(h2d_times);
      float med_kernel = median(kernel_times);
      float med_d2h = median(d2h_times);
      float med_e2e = med_h2d + med_kernel + med_d2h;

      float kernel_speedup = (med_kernel > 0) ? med_cpu / med_kernel : 0.0f;
      float e2e_speedup = (med_e2e > 0) ? med_cpu / med_e2e : 0.0f;

      csv.write_row("coordinate_embedding", B, N, D, F, input_count,
                    output_count, med_cpu, med_h2d, med_kernel, med_d2h,
                    med_e2e, kernel_speedup, e2e_speedup, -1);

      printf("CPU: %.3f ms  |  H2D: %.3f  Kernel: %.3f  D2H: %.3f"
             "E2E: %.3f ms  |  Speedup: kernel=%.1fx  e2e=%.1fx\n",
             med_cpu, med_h2d, med_kernel, med_d2h, med_e2e, kernel_speedup,
             e2e_speedup);

      bool ok = check_parity(h_cpu_out.data(), h_gpu_out.data(), output_count);
      if (!ok) {
        fprintf(stderr, "PARITY FAILED for N=%d F=%d\n", N, F);
        CUDA_CHECK(cudaFree(d_input));
        CUDA_CHECK(cudaFree(d_output));
        return 1;
      }
    }
  }

  printf("\nResults written to benchmarks/results/coordinate_embedding.csv\n");

  CUDA_CHECK(cudaFree(d_input));
  CUDA_CHECK(cudaFree(d_output));

  return 0;
}
