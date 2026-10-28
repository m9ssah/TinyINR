/*
CPU ref test for attention

construction (dims B=2, N=3, L=4, input_dim=5, D_model=8, H=2, d_k=4):

  latents[b, l, :] is c_b at dims l and l+4, zero elsewhere.
  w_KV is identity in the K block and 2*identity in the V block, so
  K[l] = c_b * e_l and V[l] = 2*c_b * e_l + b_V, per head.

  Head h therefore sees k_l = c_b * e_l in its 4-dim slice, so
    scores[l] = (q_h . k_l) / sqrt(d_k) = c_b * q_h[l] / 2
  and, because sum_l p[l] == 1,
    heads[h*4 + j] = 2*c_b*p[j] + b_V[h*4 + j].

  w_O is a cyclic shift plus a half-weight diagonal, so
    output[d] = heads[(d-1) mod 8] + 0.5*heads[d] + b_O[d].
  The shift crosses the head boundary, which is what proves heads actually mix.
*/

/*
Index letters:
b = sample,
n = query point,
l = latent,
d = model channel(0..7),
h = head (0 or 1),
j = channel within one head (0..3, so d = h*4 + j),
i = input feature channel (0..4).

How the rigging works -------------------------------------------------

1. Each latent is a "one-hot" row: latent l holds the value c_b at channel l
   AND at channel l+4, and zero everywhere else. c_b is 1.0 for sample 0 and
   1.5 for sample 1, so the two samples are not interchangeable.

     latent 0 = (c,0,0,0, c,0,0,0)     head 0 reads channels 0-3 -> (c,0,0,0)
     latent 1 = (0,c,0,0, 0,c,0,0)     head 1 reads channels 4-7 -> (c,0,0,0)
     latent 2 = (0,0,c,0, 0,0,c,0)
     latent 3 = (0,0,0,c, 0,0,0,c)

   Two hot channels, not one, so BOTH heads get a live key. Head h's view of
   latent l is c_b times the unit vector pointing along axis l.

2. w_KV is [8, 16]: the fused K|V projection. Its left half (columns 0-7) is
   the identity, so K = latents unchanged. Its right half (columns 8-15) is
   2*identity, so V = 2*latents. K and V must differ, otherwise swapping the
   two halves of the fused projection would be undetectable.

3. Because K[l] is one-hot, the score dot product collapses to a single
   multiply instead of a 4-term sum:

     scores[l] = (q_h . k_l) / sqrt(d_k) = c_b * q_h[l] / 2

   where q_h is head h's 4-wide slice of Q, and sqrt(d_k) = 2.

4. V[l] is 2*c_b at channel l plus the bias b_V (the V half of b_KV), so the
   weighted sum telescopes. Writing p[j] for the softmax probability of
   latent j, and using the fact that probabilities sum to 1:

     heads[h*4 + j] = sum_l p[l] * V[l][h*4 + j]
                    = 2*c_b*p[j] + b_V[h*4 + j]

   That last step depends on sum(p) == 1, so it also tests normalization.

5. w_O is [8, 8] with two overlaid patterns: 1.0 one step right of the
   diagonal (wrapping at the end), and 0.5 on the diagonal itself.

            d= 0    1    2    3  |  4    5    6    7
       i=0 [ 0.5  1.0    .    .  |   .    .    .    .  ]
       i=1 [   .  0.5  1.0    .  |   .    .    .    .  ]
       i=2 [   .    .  0.5  1.0  |   .    .    .    .  ]
       i=3 [   .    .    .  0.5  | 1.0    .    .    .  ]  <- crosses to head 1
       i=4 [   .    .    .    .  | 0.5  1.0    .    .  ]
       i=5 [   .    .    .    .  |   .  0.5  1.0    .  ]
       i=6 [   .    .    .    .  |   .    .  0.5  1.0  ]
       i=7 [ 1.0    .    .    .  |   .    .    .  0.5  ]  <- wraps to column 0

   Summing down column d gives:

     output[d] = heads[(d-1) mod 8] + 0.5*heads[d] + b_O[d]

   The off-diagonal 1.0 in row 3 lands in column 4, which belongs to head 1.
   That crossing is the proof that the output projection actually mixes heads
   (CLAUDE.md decision 4); delete w_O and this test fails. The 0.5 diagonal
   gives every column a SECOND non-zero, so the output projection has to be a
   real dot product rather than a permutation of channels.

--- Worked example (sample 0, point 0), verified against the implementation --

  features = (-0.20, 0.10, 0.40, 0.70, 1.00)
  Q        = (-0.15, 0.20, 0.55, 0.90, | 1.25, 0.10, 0.45, 0.80)
  head 0 scores = Q[0..3] / 2 = (-0.0750, 0.1000, 0.2750, 0.4500)
  head 0 probs  = (0.1887, 0.2247, 0.2677, 0.3189)
  heads    = (0.6273, 0.9495, 1.2854, 1.6378, | 1.9097, 1.8712, 2.1922, 2.5268)
  output   = (2.7405, 0.8020, 1.0922, 1.4043, | 1.6927, 1.7453, 1.6673, 1.9556)

  output[4] = heads[3] + 0.5*heads[4] + b_O[4]
            = 1.6378  + 0.9549        - 0.9     = 1.6927
  i.e. head 0's last channel really does reach head 1's first output channel.
  */

#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "ops/attention.h"

namespace {

constexpr int64_t kBatch = 2;
constexpr int64_t kPoints = 3;
constexpr int64_t kLatents = 4;
constexpr int64_t kInputDim = 5;
constexpr int64_t kModelDim = 8;
constexpr int64_t kHeads = 2;
constexpr int64_t kHeadDim = kModelDim / kHeads;

void assertNear(float actual, float expected) {
  assert(std::fabs(actual - expected) < 1e-6f);
}

float latentScale(int64_t b) { return b == 0 ? 1.0f : 1.5f; }
float biasQ(int64_t d) { return 0.05f * static_cast<float>(d + 1); }
float biasV(int64_t d) { return 0.25f * static_cast<float>(d + 1); }
float biasO(int64_t d) { return 0.1f - 0.2f * static_cast<float>(d + 1); }

float featureAt(int64_t b, int64_t n, int64_t i) {
  return 0.3f * static_cast<float>(i + 1) - 0.5f * static_cast<float>(n + 1) +
         0.7f * static_cast<float>(b);
}

AttentionConfig makeConfig() {
  AttentionConfig config;
  config.input_dim = kInputDim;
  config.latent_dim = kModelDim;
  config.num_heads = kHeads;
  config.num_latents = kLatents;
  return config;
}

AttentionWeights makeWeights() {
  Tensor w_Q({kInputDim, kModelDim});
  for (int64_t i = 0; i < kInputDim; i++) {
    w_Q.at({i, i}) = 1.0f;
  }
  w_Q.at({0, 5}) = 1.0f;
  w_Q.at({1, 6}) = 1.0f;
  w_Q.at({2, 7}) = 1.0f;

  Tensor b_Q({kModelDim});
  for (int64_t d = 0; d < kModelDim; d++) {
    b_Q.at({d}) = biasQ(d);
  }

  Tensor w_KV({kModelDim, 2 * kModelDim});
  for (int64_t i = 0; i < kModelDim; i++) {
    w_KV.at({i, i}) = 1.0f; // K = latents
    w_KV.at({i, kModelDim + i}) =
        2.0f; // V = 2 * latents (mult by 2 to catch a swap)
  }

  // The K half of b_KV is deliberately left at zero: a key bias adds q.b_K to
  // every score in a row equally, and softmax is invariant to a constant row
  // shift, so it can never change the output. The V half is non-zero, which is
  // what catches the two halves of the fused projection being swapped.
  Tensor b_KV({2 * kModelDim});
  for (int64_t d = 0; d < kModelDim; d++) {
    b_KV.at({kModelDim + d}) = biasV(d);
  }

  Tensor w_O({kModelDim, kModelDim});
  for (int64_t i = 0; i < kModelDim; ++i) {
    w_O.at({i, (i + 1) % kModelDim}) = 1.0f; // cyclic shift
    w_O.at({i, i}) = 0.5f;                   // diagonal half-weight
  }

  Tensor b_O({kModelDim});
  for (int64_t d = 0; d < kModelDim; ++d) {
    b_O.at({d}) = biasO(d);
  }

  return AttentionWeights{w_Q, b_Q, w_KV, b_KV, w_O, b_O};
}

Tensor makeLatents() {
  Tensor latents({kBatch, kLatents, kModelDim});
  for (int64_t b = 0; b < kBatch; ++b) {
    for (int64_t l = 0; l < kLatents; ++l) {
      latents.at({b, l, l}) = latentScale(b);
      latents.at({b, l, l + kHeadDim}) = latentScale(b);
    }
  }
  return latents;
}

Tensor makeFeatures(float gain) {
  Tensor features({kBatch, kPoints, kInputDim});
  for (int64_t b = 0; b < kBatch; ++b) {
    for (int64_t n = 0; n < kPoints; ++n) {
      for (int64_t i = 0; i < kInputDim; ++i) {
        features.at({b, n, i}) = gain * featureAt(b, n, i);
      }
    }
  }
  return features;
}

std::vector<float> expectedQueries(const Tensor &features, int64_t b,
                                   int64_t n) {
  std::vector<float> q(static_cast<size_t>(kModelDim));
  for (int64_t d = 0; d < kInputDim; ++d) {
    q[static_cast<size_t>(d)] = features.at({b, n, d});
  }
  q[5] = features.at({b, n, 0});
  q[6] = features.at({b, n, 1});
  q[7] = features.at({b, n, 2});
  for (int64_t d = 0; d < kModelDim; ++d) {
    q[static_cast<size_t>(d)] += biasQ(d);
  }
  return q;
}

// saturated == true asks for the infinite-score limit of softmax (one-hot on
// the row max), which is what a numerically stable implementation must
// converge to and what an unstable one turns into NaN.
std::vector<float> expectedOutput(const Tensor &features, int64_t b, int64_t n,
                                  bool saturated) {
  const std::vector<float> q = expectedQueries(features, b, n);
  const float c = latentScale(b);
  const float scale = 1.0f / std::sqrt(static_cast<float>(kHeadDim));

  std::vector<float> heads(static_cast<size_t>(kModelDim));
  for (int64_t h = 0; h < kHeads; ++h) {
    std::vector<float> scores(static_cast<size_t>(kHeadDim));
    for (int64_t j = 0; j < kHeadDim; ++j) {
      scores[static_cast<size_t>(j)] =
          scale * c * q[static_cast<size_t>(h * kHeadDim + j)];
    }

    std::vector<float> probs(static_cast<size_t>(kHeadDim), 0.0f);
    if (saturated) {
      size_t best = 0;
      for (size_t j = 1; j < probs.size(); ++j) {
        if (scores[j] > scores[best]) {
          best = j;
        }
      }
      probs[best] = 1.0f;
    } else {
      float total = 0.0f;
      for (size_t j = 0; j < probs.size(); ++j) {
        probs[j] = std::exp(scores[j]);
        total += probs[j];
      }
      for (size_t j = 0; j < probs.size(); ++j) {
        probs[j] /= total;
      }
    }

    for (int64_t j = 0; j < kHeadDim; ++j) {
      heads[static_cast<size_t>(h * kHeadDim + j)] =
          2.0f * c * probs[static_cast<size_t>(j)] + biasV(h * kHeadDim + j);
    }
  }

  std::vector<float> output(static_cast<size_t>(kModelDim));
  for (int64_t d = 0; d < kModelDim; ++d) {
    const size_t shifted = static_cast<size_t>((d + kModelDim - 1) % kModelDim);
    output[static_cast<size_t>(d)] =
        heads[shifted] + 0.5f * heads[static_cast<size_t>(d)] + biasO(d);
  }
  return output;
}

void expectThrows(const AttentionWeights &weights,
                  const AttentionInputs &inputs, const AttentionConfig &config,
                  const char *what) {
  bool threw = false;
  try {
    (void)attention(weights, inputs, config);
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  if (!threw) {
    std::cerr << "expected a throw for: " << what << "\n";
  }
  assert(threw);
}

} // namespace

int main() {
  const AttentionConfig config = makeConfig();
  const AttentionWeights weights = makeWeights();
  const Tensor latents = makeLatents();
  const Tensor features = makeFeatures(1.0f);

  // 1. full forward against the closed form above
  const Tensor output = attention(weights, {features, latents}, config);
  assert(output.shape() == std::vector<int64_t>({kBatch, kPoints, kModelDim}));
  for (int64_t b = 0; b < kBatch; ++b) {
    for (int64_t n = 0; n < kPoints; ++n) {
      const std::vector<float> expected = expectedOutput(features, b, n, false);
      for (int64_t d = 0; d < kModelDim; ++d) {
        assertNear(output.at({b, n, d}), expected[static_cast<size_t>(d)]);
      }
    }
  }

  // 2. Attention is permutation-invariant over the latent bank: reordering the
  // (K, V) pairs cannot change the result. Catches an l/d index swap.
  {
    const int64_t permutation[kLatents] = {2, 0, 3, 1};
    Tensor permuted({kBatch, kLatents, kModelDim});
    for (int64_t b = 0; b < kBatch; ++b) {
      for (int64_t l = 0; l < kLatents; ++l) {
        for (int64_t d = 0; d < kModelDim; ++d) {
          permuted.at({b, permutation[l], d}) = latents.at({b, l, d});
        }
      }
    }
    const Tensor reordered = attention(weights, {features, permuted}, config);
    for (int64_t i = 0; i < output.numel(); ++i) {
      assertNear(reordered.data()[static_cast<size_t>(i)],
                 output.data()[static_cast<size_t>(i)]);
    }
  }

  // 3. Samples must not leak into each other: sample 1 run alone has to
  // reproduce the b=1 slice of the batched run. Catches a batch-stride bug.
  {
    Tensor lone_features({1, kPoints, kInputDim});
    Tensor lone_latents({1, kLatents, kModelDim});
    for (int64_t n = 0; n < kPoints; ++n) {
      for (int64_t i = 0; i < kInputDim; ++i) {
        lone_features.at({0, n, i}) = features.at({1, n, i});
      }
    }
    for (int64_t l = 0; l < kLatents; ++l) {
      for (int64_t d = 0; d < kModelDim; ++d) {
        lone_latents.at({0, l, d}) = latents.at({1, l, d});
      }
    }
    const Tensor lone =
        attention(weights, {lone_features, lone_latents}, config);
    for (int64_t n = 0; n < kPoints; ++n) {
      for (int64_t d = 0; d < kModelDim; ++d) {
        assertNear(lone.at({0, n, d}), output.at({1, n, d}));
      }
    }
  }

  // 4. Numerical stability: scores around 5e3 overflow exp() unless the row max
  // is subtracted first. Output must stay finite and land on the one-hot limit.
  {
    const Tensor large = makeFeatures(1.0e4f);
    const Tensor saturated = attention(weights, {large, latents}, config);
    for (int64_t b = 0; b < kBatch; ++b) {
      for (int64_t n = 0; n < kPoints; ++n) {
        const std::vector<float> expected = expectedOutput(large, b, n, true);
        for (int64_t d = 0; d < kModelDim; ++d) {
          assert(std::isfinite(saturated.at({b, n, d})));
          assertNear(saturated.at({b, n, d}), expected[static_cast<size_t>(d)]);
        }
      }
    }
  }

  // 5. Validation: every rejection branch.
  {
    const AttentionInputs inputs{features, latents};

    AttentionConfig indivisible = config;
    indivisible.num_heads = 3; // 8 % 3 != 0
    expectThrows(weights, inputs, indivisible, "latent_dim % num_heads != 0");

    AttentionConfig no_heads = config;
    no_heads.num_heads = 0;
    expectThrows(weights, inputs, no_heads, "num_heads == 0");

    AttentionConfig wrong_input_dim = config;
    wrong_input_dim.input_dim = kInputDim - 1;
    expectThrows(weights, inputs, wrong_input_dim, "input_dim mismatch");

    AttentionConfig wrong_latent_count = config;
    wrong_latent_count.num_latents = kLatents - 1;
    expectThrows(weights, inputs, wrong_latent_count, "num_latents mismatch");

    const AttentionInputs flat{features.reshape({kBatch * kPoints, kInputDim}),
                               latents};
    expectThrows(weights, flat, config, "features is not 3-D");

    Tensor short_batch({kBatch - 1, kLatents, kModelDim});
    expectThrows(weights, {features, short_batch}, config,
                 "latents batch size mismatch");

    AttentionWeights bad_w_Q = weights;
    bad_w_Q.w_Q = Tensor({kInputDim - 1, kModelDim});
    expectThrows(bad_w_Q, inputs, config, "mis-shaped w_Q");

    AttentionWeights bad_b_Q = weights;
    bad_b_Q.b_Q = Tensor({kModelDim - 1});
    expectThrows(bad_b_Q, inputs, config, "mis-shaped b_Q");

    AttentionWeights bad_w_KV = weights;
    bad_w_KV.w_KV = Tensor({kModelDim, kModelDim}); // unfused width
    expectThrows(bad_w_KV, inputs, config, "mis-shaped w_KV");

    AttentionWeights bad_b_KV = weights;
    bad_b_KV.b_KV = Tensor({kModelDim});
    expectThrows(bad_b_KV, inputs, config, "mis-shaped b_KV");

    AttentionWeights bad_w_O = weights;
    bad_w_O.w_O = Tensor({kModelDim, kHeadDim});
    expectThrows(bad_w_O, inputs, config, "mis-shaped w_O");

    AttentionWeights bad_b_O = weights;
    bad_b_O.b_O = Tensor({kHeadDim});
    expectThrows(bad_b_O, inputs, config, "mis-shaped b_O");
  }

  std::cout << "test_attention PASS\n";
  return 0;
}
