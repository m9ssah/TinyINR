#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "ops/fourier_embedding.h"
#include "training/sampled_train_step.h"

namespace {
using Clock = std::chrono::steady_clock;

struct Options {
  std::string mode = "cicfm", sampling = "adaptive", csv = "adaptive_sampling.csv";
  int points = 256, samples = 64, steps = 20, hidden = 16, frequencies = 8;
  int eval_every = 5, eval_probes = 3;
  float learning_rate = 0.001f, eta = 0.10f, beta = 0.90f;
  uint32_t model_seed = 17, sampling_seed = 42, noise_seed = 123, eval_seed = 100000;
};

void usage() {
  std::cout << "CPU adaptive sampling demo (uncorrected training loss)\n"
      "Options: --mode mse|cicfm --sampling uniform|adaptive\n"
      "  --points 256 --samples 64 --steps 20 --hidden 16 --frequencies 8\n"
      "  --learning-rate 0.001 --eta 0.10 --beta 0.90\n"
      "  --model-seed 17 --sampling-seed 42 --noise-seed 123 --eval-seed 100000\n"
      "  --eval-every 5 --eval-probes 3 --csv adaptive_sampling.csv\n"
      "The CSV path is overwritten. Evaluation includes step 0 and the final step.\n";
}

uint64_t integer(const std::string &value, uint64_t maximum) {
  if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos) {
    throw std::invalid_argument("expected an unsigned integer: " + value);
  }
  const auto result = std::stoull(value);
  if (result > maximum) {
    throw std::invalid_argument("integer out of range: " + value);
  }
  return result;
}

float real(const std::string &value) {
  size_t consumed = 0;
  const float result = std::stof(value, &consumed);
  if (consumed != value.size() || !std::isfinite(result)) {
    throw std::invalid_argument("expected a finite number: " + value);
  }
  return result;
}

Options parse(int argc, char **argv) {
  Options o;
  std::set<std::string> seen;
  for (int i = 1; i < argc; ++i) {
    const std::string key = argv[i];
    if (!seen.insert(key).second || i + 1 == argc) {
      throw std::invalid_argument("duplicate option or missing value: " + key);
    }
    const std::string value = argv[++i];
    const auto number = [&]() {
      return static_cast<int>(integer(value, std::numeric_limits<int>::max()));
    };
    const auto seed = [&]() {
      return static_cast<uint32_t>(integer(value, std::numeric_limits<uint32_t>::max()));
    };
    if (key == "--mode") o.mode = value;
    else if (key == "--sampling") o.sampling = value;
    else if (key == "--csv") o.csv = value;
    else if (key == "--points") o.points = number();
    else if (key == "--samples") o.samples = number();
    else if (key == "--steps") o.steps = number();
    else if (key == "--hidden") o.hidden = number();
    else if (key == "--frequencies") o.frequencies = number();
    else if (key == "--eval-every") o.eval_every = number();
    else if (key == "--eval-probes") o.eval_probes = number();
    else if (key == "--learning-rate") o.learning_rate = real(value);
    else if (key == "--eta") o.eta = real(value);
    else if (key == "--beta") o.beta = real(value);
    else if (key == "--model-seed") o.model_seed = seed();
    else if (key == "--sampling-seed") o.sampling_seed = seed();
    else if (key == "--noise-seed") o.noise_seed = seed();
    else if (key == "--eval-seed") o.eval_seed = seed();
    else throw std::invalid_argument("unknown option: " + key);
  }
  if ((o.mode != "mse" && o.mode != "cicfm") ||
      (o.sampling != "uniform" && o.sampling != "adaptive") || o.csv.empty() ||
      o.points <= 0 || o.samples <= 0 || o.samples > o.points || o.steps <= 0 ||
      o.hidden <= 0 || o.frequencies <= 0 || o.eval_every <= 0 || o.eval_probes <= 0 ||
      o.learning_rate <= 0 || o.eta <= 0 || o.eta > 1 || o.beta < 0 || o.beta >= 1) {
    throw std::invalid_argument("invalid demo configuration (see --help)");
  }
  const int64_t features = 2 + 4 * static_cast<int64_t>(o.frequencies);
  if (o.frequencies >= std::numeric_limits<float>::max_exponent ||
      features > std::numeric_limits<int>::max() / o.points) {
    throw std::invalid_argument("full-grid features exceed CPU embedding limits");
  }
  const uint64_t max_seed = std::numeric_limits<uint32_t>::max();
  if (static_cast<uint64_t>(o.noise_seed) + 2ULL * o.steps - 1 > max_seed ||
      static_cast<uint64_t>(o.eval_seed) + 2ULL * o.eval_probes - 1 > max_seed) {
    throw std::invalid_argument("noise/time seed schedule would wrap");
  }
  return o;
}

CoordinateBatch syntheticImage(int points) {
  const int width = static_cast<int>(std::ceil(std::sqrt(static_cast<double>(points))));
  const int height = static_cast<int>((static_cast<int64_t>(points) + width - 1) / width);
  Tensor coordinates({1, points, 2}), values({1, points, 3});
  for (int n = 0; n < points; ++n) {
    const int row = n / width, col = n % width;
    const float x = width == 1 ? 0.5f : static_cast<float>(col) / (width - 1);
    const float y = height == 1 ? 0.5f : static_cast<float>(row) / (height - 1);
    coordinates.at({0, n, 0}) = 2 * x - 1;
    coordinates.at({0, n, 1}) = 2 * y - 1;
    const float detail = x < 0.5f ? 0.5f : ((row + col) % 2 ? 0.1f : 0.9f);
    values.at({0, n, 0}) = x;
    values.at({0, n, 1}) = y;
    values.at({0, n, 2}) = detail;
  }
  return CoordinateBatch(coordinates, values);
}

using Probes = std::vector<std::pair<Tensor, Tensor>>;

Probes makeProbes(const CoordinateBatch &batch, const Options &o) {
  Tensor features = fourierEmbedding(batch.coordinates().reshape({o.points, 2}),
                                     o.frequencies).reshape({1, o.points, 2 + 4LL * o.frequencies});
  Probes probes;
  if (o.mode == "mse") {
    probes.emplace_back(std::move(features), batch.values());
  } else {
    for (int p = 0; p < o.eval_probes; ++p) {
      const uint32_t seed = static_cast<uint32_t>(o.eval_seed + 2ULL * p);
      Tensor z0 = sampleUniform(batch.values().shape(), 0, 1, seed);
      Tensor t = sampleUniform({1, o.points, 1}, 0, 1, seed + 1);
      auto probe = makeCicfmBatch(features, batch.values(), z0, t);
      probes.emplace_back(std::move(probe.input), std::move(probe.target_velocity));
    }
  }
  return probes;
}

double evaluate(const Mlp &model, const Probes &probes) {
  double sum = 0;
  for (const auto &probe : probes) {
    const auto prediction = mlpForward(model, probe.first);
    const float loss = mseLoss(prediction.output, probe.second);
    if (!std::isfinite(loss)) throw std::runtime_error("non-finite evaluation loss");
    sum += loss;
  }
  return sum / probes.size();
}

void run(const Options &o) {
  const auto batch = syntheticImage(o.points);
  const auto probes = makeProbes(batch, o);
  const LossMode mode = o.mode == "mse" ? LossMode::MSE : LossMode::CICFM;
  const int64_t feature_dim = mseInputDim(2, o.frequencies);
  MlpConfig model_config{feature_dim + (mode == LossMode::CICFM ? 4 : 0),
                         o.hidden, 3, 3, ActivationKind::SiLU};
  Mlp model = createMlp(model_config, o.model_seed);
  Tensor priorities({1, o.points, 1});
  std::fill(priorities.data(), priorities.data() + priorities.numel(), 1.0f);
  SamplingConfig sampling;
  sampling.eta = o.sampling == "uniform" ? 1.0f : o.eta;
  std::mt19937 rng(o.sampling_seed);
  TrainStepConfig config{mode, 2, 3, o.frequencies, o.learning_rate, false, o.noise_seed};
  std::vector<bool> covered(static_cast<size_t>(o.points), false);
  int coverage = 0;
  std::ofstream csv;
  csv.exceptions(std::ios::failbit | std::ios::badbit);
  csv.open(o.csv);
  csv << std::setprecision(17);
  csv << "step,mode,sampling,points,samples,steps,hidden,frequencies,learning_rate,"
      "eta,epsilon,beta,model_seed,sampling_seed,noise_seed,step_noise_seed,eval_seed,"
      "eval_every,eval_probes,build_type,sampled_loss,eval_metric,eval_loss,unique_points,"
      "coverage_count,coverage_fraction,duplicate_rate,probability_min,probability_max,"
      "priority_min,priority_max,all_finite,sampling_ms,gather_ms,embedding_ms,train_ms,"
      "residual_ms,update_ms,step_ms,eval_ms\n";
  const char *build_type = TINYINR_BUILD_TYPE;
  const char *metric = mode == LossMode::MSE ? "rgb_mse" : "velocity_mse";
  double last_eval = 0;
  for (int64_t step = 0; step <= o.steps; ++step) {
    const auto probabilities = coordinateSamplingProbabilities(priorities, sampling);
    const auto probability_range = std::minmax_element(probabilities.begin(), probabilities.end());
    SampledTrainTimings timings;
    float sampled_loss = 0;
    int unique = 0;
    if (step > 0) {
      config.seed = static_cast<uint32_t>(o.noise_seed + 2ULL * (step - 1));
      const auto result = sampledTrainStep(model, batch, o.samples, config, priorities,
                                           sampling, rng, o.beta, &timings);
      sampled_loss = result.training.loss;
      if (!std::isfinite(sampled_loss) || !tensorStats(result.residuals).all_finite ||
          !tensorStats(priorities).all_finite) {
        throw std::runtime_error("non-finite training diagnostics");
      }
      std::vector<bool> seen(static_cast<size_t>(o.points), false);
      for (int64_t index : result.selection.indices) {
        const size_t i = static_cast<size_t>(index);
        if (!seen[i]) { seen[i] = true; ++unique; }
        if (!covered[i]) { covered[i] = true; ++coverage; }
      }
    }
    const bool evaluated = step == 0 || step % o.eval_every == 0 || step == o.steps;
    double eval_ms = 0;
    if (evaluated) {
      const auto start = Clock::now();
      last_eval = evaluate(model, probes);
      eval_ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    }
    const auto priority_range = std::minmax_element(priorities.data(),
                                                   priorities.data() + priorities.numel());
    csv << step << ',' << o.mode << ',' << o.sampling << ',' << o.points << ','
        << o.samples << ',' << o.steps << ',' << o.hidden << ',' << o.frequencies << ','
        << o.learning_rate << ',' << sampling.eta << ',' << sampling.epsilon << ','
        << o.beta << ',' << o.model_seed << ',' << o.sampling_seed << ',' << o.noise_seed << ',';
    if (step > 0) csv << config.seed;
    csv << ',' << o.eval_seed << ',' << o.eval_every << ',' << probes.size() << ',' << build_type << ',';
    if (step > 0) csv << sampled_loss;
    csv << ',' << metric << ',';
    if (evaluated) csv << last_eval;
    csv << ',' << unique << ',' << coverage << ',' << static_cast<double>(coverage) / o.points
        << ',' << (step == 0 ? 0.0 : 1.0 - static_cast<double>(unique) / o.samples)
        << ',' << *probability_range.first << ',' << *probability_range.second
        << ',' << *priority_range.first << ',' << *priority_range.second << ",1,"
        << timings.sampling_ms << ',' << timings.gather_ms << ',' << timings.embedding_ms
        << ',' << timings.train_ms << ',' << timings.residual_ms << ',' << timings.update_ms
        << ',' << timings.step_ms << ',' << eval_ms << '\n';
  }
  csv.close();
  std::cout << o.mode << '/' << o.sampling << ": final " << metric << '=' << last_eval
            << ", coverage=" << coverage << '/' << o.points << "; CSV: " << o.csv << '\n';
}
} // namespace

int main(int argc, char **argv) {
  try {
    if (argc == 2 && std::string(argv[1]) == "--help") { usage(); return 0; }
    run(parse(argc, argv));
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "train_adaptive_sampling: " << error.what() << '\n';
    return 1;
  }
}
