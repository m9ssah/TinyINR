#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "model/mlp.h"
#include "training/cicfm.h"
#include "training/loss.h"
#include "training/train_step.h"

#include "kernel/train_step.cuh"
#include "bench_utils.cuh"

static const int B = 1;
static const int D = 2;
static const int F = 8;
static const int CHANNELS = 3;
static const int NUM_TRIALS = 20;
static const float LR = 0.01f;

static const int N_VALUES[] = {1024, 16384, 65536};
static const int count_N = sizeof(N_VALUES) / sizeof(N_VALUES[0]);
static const LossMode LOSS_MODES[] = {LossMode::MSE, LossMode::CICFM};

static float mean_abs(const Tensor &a, const Tensor &b) {
  double sum = 0.0;
  for (int64_t i = 0; i < a.numel(); ++i) {
    sum += std::fabs(a.data()[i] - b.data()[i]);
  }
  return static_cast<float>(sum / a.numel());
}

static float mean_of(const Tensor &a) {
  double sum = 0.0;
  for (int64_t i = 0; i < a.numel(); ++i) {
    sum += a.data()[i];
  }
  return static_cast<float>(sum / a.numel());
}

int main() {
  printf("Train Step Benchmark\n");

  cuda_warmup();

  L2Flusher flusher;

  TrainCsvWriter csv("benchmarks/results/train_step.csv");

  for (int n = 0; n < count_N; n++) {
    const int N = N_VALUES[n];
    const int rows = B * N;

    for (LossMode mode : LOSS_MODES) {
      const bool cicfm = (mode == LossMode::CICFM);
      const char *mode_name = cicfm ? "CICFM" : "MSE";
      const int feature_dim = static_cast<int>(mseInputDim(D, F)); // 34
      const int input_dim =
          cicfm ? static_cast<int>(cicfmInputDim(D, CHANNELS, F)) : feature_dim;
      printf("N=%d mode=%s\n", N, mode_name);

      const MlpConfig config = makeBaselineMlpConfig(input_dim, CHANNELS);
      Mlp gpu_model = createMlp(config, 42);
      Mlp cpu_model = createMlp(config, 42);

      Tensor features = sampleUniform({B, N, feature_dim}, -1.0f, 1.0f, 7);
      Tensor targets = sampleUniform({B, N, CHANNELS}, 0.0f, 1.0f, 8);
      Tensor z0 = sampleUniform({B, N, CHANNELS}, -1.0f, 1.0f, 9);
      Tensor t = sampleUniform({B, N, 1}, 0.0f, 1.0f, 10);

      // CICFM input statistics required by the benchmark contract.
      const float stat_mean_t = cicfm ? mean_of(t) : NAN;
      const float stat_mean_vel = cicfm ? mean_abs(targets, z0) : NAN;

      GpuTrainContext ctx =
          createGpuTrainContext(gpu_model, config, rows, feature_dim, CHANNELS);
      gpuTrainStep(ctx, mode, features, targets, cicfm ? &z0 : nullptr,
                   cicfm ? &t : nullptr, LR);

      std::vector<float> assembly, forward, loss_ms, backward, step;
      std::vector<float> losses;
      bool all_finite = true;
      for (int trial = 0; trial < NUM_TRIALS; trial++) {
        flusher.flush();
        StageTimings tm;
        float loss =
            gpuTrainStep(ctx, mode, features, targets, cicfm ? &z0 : nullptr,
                         cicfm ? &t : nullptr, LR, &tm);
        assembly.push_back(tm.assembly_ms);
        forward.push_back(tm.forward_ms);
        loss_ms.push_back(tm.loss_ms);
        backward.push_back(tm.backward_ms);
        step.push_back(tm.step_ms);
        losses.push_back(loss);
        all_finite = all_finite && std::isfinite(loss);
      }
      float loss_sum = 0.0f;
      for (float l : losses)
        loss_sum += l;

      csv.write_row("train_step", mode_name, "gpu", B, N, feature_dim,
                    input_dim, CHANNELS, F, median(forward), median(loss_ms),
                    median(backward), median(step),
                    loss_sum / static_cast<float>(NUM_TRIALS), all_finite,
                    cicfm ? median(assembly) : NAN, stat_mean_t, stat_mean_vel,
                    NUM_TRIALS);
      freeGpuTrainContext(ctx);

      const int cpu_trials = (N > 16384) ? 1 : 3;
      TrainStepConfig cpu_config{mode,       D,  CHANNELS,
                                 F,          LR, /*deterministic_cicfm=*/false,
                                 /*seed=*/42};
      CpuTimer cpu_timer;
      std::vector<float> cpu_step;
      float cpu_loss = 0.0f;
      (void)trainStep(cpu_model, features, targets, cpu_config);
      for (int trial = 0; trial < cpu_trials; trial++) {
        cpu_timer.start();
        TrainStepResult r = trainStep(cpu_model, features, targets, cpu_config);
        cpu_timer.stop();
        cpu_step.push_back(cpu_timer.elapsed_ms());
        cpu_loss = r.loss;
      }
      csv.write_row("train_step", mode_name, "cpu", B, N, feature_dim,
                    input_dim, CHANNELS, F, NAN, NAN, NAN, median(cpu_step),
                    cpu_loss, std::isfinite(cpu_loss), NAN, stat_mean_t,
                    stat_mean_vel, cpu_trials);
    }
  }
  printf("Results written to benchmarks/results/train_step.csv\n");
  return 0;
}
