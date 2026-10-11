#pragma once

#include <cstdint>
#include <random>
#include <vector>

#include "tensor.h"

struct SamplingConfig {
  float eta = 0.10f; // Uniform exploration fraction; 1 selects uniform sampling.
  float epsilon = 1e-6f;
};

struct SampleSelection {
  int64_t batch_size;
  int64_t num_points; // Original points per example.
  int64_t sample_count;
  std::vector<int64_t> indices;
  std::vector<double> probabilities;
};

// Priorities must be finite, nonnegative, and shaped [B,N,1]. Returns B*N
// probabilities in batch-major order, normalized separately for each example.
// Requires finite 0 < eta <= 1 and finite epsilon > 0. Invalid inputs throw
// std::invalid_argument; unrepresentable buffer sizes throw std::length_error.
std::vector<double> coordinateSamplingProbabilities(
    const Tensor &priorities, const SamplingConfig &config);

// Draws with replacement, requiring 1 <= sample_count <= N. Entry b*K+k holds
// an original point index local to example b and its probability at draw time.
// Validates all inputs before consuming the caller-owned RNG. Reproducibility
// is scoped to the same toolchain; no cross-library draw sequence is promised.
SampleSelection sampleCoordinateIndices(const Tensor &priorities,
                                        int64_t sample_count,
                                        const SamplingConfig &config,
                                        std::mt19937 &generator);

// Average duplicate observations, then apply one EMA update per original point:
// new = beta*old + (1-beta)*mean_residual. Unobserved points stay unchanged.
// Residuals must be finite, nonnegative [B,K,1]; beta must be finite in [0,1).
// Validates the entire selection and priorities before committing any changes.
void updateCoordinatePriorities(Tensor &priorities,
                                const SampleSelection &selection,
                                const Tensor &residuals, float beta = 0.90f);
