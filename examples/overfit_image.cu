// usage: overfit_image <input.ppm> [steps] [lr] [log_every] [eval_every]
//                      [out_prefix] [num_frequencies] [seed]

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#include "io/ppm.h"
#include "kernel/cuda_utils.cuh"
#include "kernel/train_step.cuh"
#include "model/mlp.h"
#include "ops/coordinate_batch.h"
#include "ops/fourier_embedding.h"
#include "optim/adamw.cuh"

namespace {

struct Options {
  std::string path;
  int steps = 2000;
  float lr = 1e-3f;
  int log_every = 50;
  int eval_every = 250;
  std::string out_prefix = "overfit";
  int num_frequencies = 8;
  uint32_t seed = 42;
};

Options parseArgs(int argc, char **argv) {
  if (argc < 2) {
    throw std::invalid_argument(
        "usage: overfit_image <input.ppm> [steps] [lr] [log_every] "
        "[eval_every] [out_prefix] [num_frequencies] [seed]");
  }
  Options o;
  o.path = argv[1];
  if (argc > 2)
    o.steps = std::atoi(argv[2]);
  if (argc > 3)
    o.lr = static_cast<float>(std::atof(argv[3]));
  if (argc > 4)
    o.log_every = std::atoi(argv[4]);
  if (argc > 5)
    o.eval_every = std::atoi(argv[5]);
  if (argc > 6)
    o.out_prefix = argv[6];
  if (argc > 7)
    o.num_frequencies = std::atoi(argv[7]);
  if (argc > 8)
    o.seed = static_cast<uint32_t>(std::strtoul(argv[8], nullptr, 10));
  if (o.steps <= 0 || o.log_every <= 0 || o.eval_every <= 0 ||
      o.num_frequencies <= 0 || !(o.lr > 0.0f)) {
    throw std::invalid_argument("steps, lr, log_every, eval_every and "
                                "num_frequencies must be positive");
  }
  return o;
}

double msSince(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now() - start)
      .count();
}

double psnrFromMse(double mse) {
  return 10.0 * std::log10(1.0 / std::max(mse, 1e-12));
}

double clampedPsnr(const std::vector<float> &prediction,
                   const std::vector<float> &target) {
  double sum = 0.0;
  for (size_t i = 0; i < target.size(); i++) {
    const double p = std::min(1.0f, std::max(0.0f, prediction[i]));
    const double d = p - target[i];
    sum += d * d;
  }
  return psnrFromMse(sum / static_cast<double>(target.size()));
}

std::string stepPath(const std::string &prefix, int step) {
  char suffix[32];
  std::snprintf(suffix, sizeof(suffix), "_step%05d.ppm", step);
  return prefix + suffix;
}

void overfitImage(const Options &o) {
  // setup:
  Image img = readPPM(o.path);
  const int rows = img.width * img.height;
  CoordinateBatch batch =
      generateCoordinateBatch(img.data, img.height, img.width, img.channels);
  Tensor features =
      fourierEmbedding(batch.coordinates().reshape({rows, batch.coord_dim()}),
                       o.num_frequencies);
  const int feature_dim = static_cast<int>(features.shape()[1]);
  MlpConfig cfg = makeBaselineMlpConfig(feature_dim, img.channels);
  Mlp model = createMlp(cfg, o.seed);
  GpuTrainContext ctx =
      createGpuTrainContext(model, cfg, rows, feature_dim, img.channels);
  GpuAdamW adam = createGpuAdamW(ctx.mlp);
  gpuUploadTrainData(ctx, features, batch.values());

  std::printf("%s: %dx%d, %d rows, feature_dim %d, %d steps, lr %g\n",
              o.path.c_str(), img.width, img.height, rows, feature_dim, o.steps,
              o.lr);

  const std::string csv_path = o.out_prefix + ".csv";
  FILE *csv = std::fopen(csv_path.c_str(), "w");
  if (!csv)
    throw std::runtime_error("could not open " + csv_path);
  std::fprintf(csv, "step,train_ms,loss,psnr,psnr_clamped\n");

  // loop:
  AdamWConfig opt;
  opt.lr = o.lr;
  Image recon = img;

  const double thresholds[] = {20.0, 25.0, 30.0};
  double time_to[] = {-1.0, -1.0, -1.0};
  double train_ms = 0.0;
  double last_psnr_clamped = 0.0;
  float last_loss = 0.0f;
  auto seg_start = std::chrono::steady_clock::now();

  for (int step = 1; step <= o.steps; step++) {
    gpuTrainStepResident(ctx, adam, opt);

    if (step % o.log_every == 0 || step == o.steps) {
      const float loss = gpuReadLoss(ctx);
      train_ms += msSince(seg_start);

      cuda_d2h(recon.data.data(), ctx.cache.output, recon.data.size());
      const double psnr_clamped = clampedPsnr(recon.data, img.data);

      std::fprintf(csv, "%d,%.3f,%.8f,%.4f,%.4f\n", step, train_ms, loss,
                   psnrFromMse(loss), psnr_clamped);
      for (int k = 0; k < 3; k++) {
        if (time_to[k] < 0.0 && psnr_clamped >= thresholds[k])
          time_to[k] = train_ms;
      }
      if (step % o.eval_every == 0 || step == o.steps) {
        writePPM(stepPath(o.out_prefix, step), recon); // clamps + rounds
        std::printf("step %5d  loss %.6f  psnr %.2f dB  %.0f ms\n", step, loss,
                    psnr_clamped, train_ms);
      }
      last_loss = loss;
      last_psnr_clamped = psnr_clamped;

      // restart after eval and file i/p, so they aren't counted as training
      seg_start = std::chrono::steady_clock::now();
    }
  }
  std::fclose(csv);

  std::printf("\nfinal: loss %.6f, psnr %.2f dB, %.2f ms/step over %d steps\n",
              last_loss, last_psnr_clamped, train_ms / o.steps, o.steps);
  for (int k = 0; k < 3; k++) {
    if (time_to[k] >= 0.0)
      std::printf("time to %.0f dB: %.0f ms\n", thresholds[k], time_to[k]);
    else
      std::printf("time to %.0f dB: not reached\n", thresholds[k]);
  }
  std::printf("wrote %s and %s_step*.ppm\n", csv_path.c_str(),
              o.out_prefix.c_str());

  freeGpuAdamW(adam);
  freeGpuTrainContext(ctx);
}

} // namespace

int main(int argc, char **argv) {
  try {
    overfitImage(parseArgs(argc, argv));
  } catch (const std::exception &e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
  }
  return 0;
}
