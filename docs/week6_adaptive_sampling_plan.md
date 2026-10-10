# Week 6: CPU adaptive coordinate sampling

Status: Commits 1-2 are complete: the contract and CPU probability/index sampler
are implemented. Gathering, training integration, and priority updates remain planned.
Baseline reviewed: `adaptive_sampling` at `b546546`.
Input: the supplied partner attention plan and this branch's source code.

Review validation: a fresh Debug CMake build of this baseline succeeded, and all
eight existing CPU test executables passed, including the full-size baseline.
CUDA was unavailable on this machine, so no GPU execution was performed. Commit 1
changes this document only; it does not implement the sampler or register CTest tests.

## Outcome

Deliver an independently usable CPU sampler and a reproducible training demo:

```text
Fixed CoordinateBatch + priorities for its original points
  -> uniform/adaptive probabilities
  -> original point indices + probabilities at draw time
  -> gather matching coordinates and values
  -> Fourier embedding
  -> existing MSE/CICFM trainStep
  -> per-point prediction error
  -> update priorities at the original indices
  -> next draw
```

Success means the loop is correct, reproducible, and measurable. It does not mean
adaptive sampling is guaranteed to beat uniform sampling. Week 6 ends with a CPU
reference; CUDA sampling, broad experiments, and attention integration are later
work. This narrows the supplied plan's ambiguous "adaptive sampling complete"
milestone to a result that can be demonstrated on the current branch.

## What exists, and what is missing

| Current file | Reuse or gap |
|---|---|
| `include/ops/coordinate_batch.h`, `src/ops/coordinate_batch.cpp` | Own aligned coordinate/value tensors. `randomSubsample` shuffles without replacement, creates its own random seed, and returns neither source indices nor probabilities. |
| `include/tensor.h`, `src/tensor.cpp` | CPU float tensors suffice for priorities/residuals. Use integer vectors for point indices; do not store indices in float tensors. |
| `src/ops/fourier_embedding.cpp` | Accepts `[N,D]`, while batches contain `[B,K,D]`. Flatten and restore dimensions at the training seam. |
| `src/ops/fourier_embedding_raw.cpp` | Produces raw coordinates followed by frequency-major sine/cosine features. Preserve this training layout. |
| `src/training/train_step.cpp` | Implements CPU MSE/CICFM forward, loss, backward, SGD. No general coordinate training driver exists. |
| `include/training/train_step.h` | `TrainStepResult.model_output` and `.loss_target` already provide everything needed for residuals. They describe the forward pass before the SGD update. |
| `src/training/cicfm.cpp` | Handles noise/time generation and interpolation. A repeated seed recreates the same random arrays. |
| `src/kernels/*` | Contains CUDA primitives but this branch has no complete GPU train-step function. Existing GPU embedding also differs from the CPU training feature contract. |
| `CMakeLists.txt` | Builds eight CPU test executables; tests are not registered with CTest. CPU sampling targets must be outside the CUDA-only block. |

Existing training tests mostly construct feature tensors directly. They do not
prove the real CoordinateBatch -> embedding -> training path. Close this gap
before claiming an adaptive training loop works.

The production train step also omits a finite check on scalar loss. Include that
small guard before SGD when wiring the loop, since finite residuals can overflow
when squared. Do not expand this into unrelated cleanup.

## Decisions to freeze before implementation

### 1. Stable identity and shapes

Start with a fixed set of examples and a fixed ordering of points. `B` is the
number of examples, `N` the original points per example, `K` draws per step, `D`
coordinate dimensions, and `C` value channels. All examples have the same N.

| Data | Shape/storage |
|---|---|
| Full coordinates / values | `[B,N,D]` / `[B,N,C]` |
| Persistent priorities | `[B,N,1]`, initially all `1` |
| Full draw distribution | Conceptually `[B,N]`, normalize each example separately |
| Selected original indices | `std::vector<int64_t>`, size `B*K`, batch-major |
| Selected draw probabilities | `std::vector<double>`, size `B*K`, same order |
| Gathered coordinates / values | `[B,K,D]` / `[B,K,C]` |
| Per-draw residuals | `[B,K,1]` |

The entry at flat offset `b*K+k` identifies original point `indices[b*K+k]` in
example b; its probability is stored at that same offset. Require positive B/N/K,
exactly B*K entries in each selection vector, and indices in `[0,N)`. Reject
incompatible shapes or sizes before drawing or gathering. Check size products
before allocating or indexing buffers.

An index is local to the original example, never to the previously sampled
subset. Priorities persist across training steps, but must be reset if examples
are reordered or replaced. Multi-image dataset identity/checkpointing is deferred.

### 2. Sampling rule

Use independent categorical draws **with replacement** for both the new uniform
baseline and adaptive mode. Duplicates are valid. Keep the existing
`randomSubsample` behavior unchanged; its no-replacement selection is a different
baseline. For the Week 6 demo, require `1 <= K <= N`, even though categorical
sampling can mathematically support larger K.

For original point i within one example:

```text
q_i = (priority_i + epsilon) / sum_j(priority_j + epsilon)
p_i = eta/N + (1-eta)*q_i
```

Frozen v1 defaults (configurable, not tuned research results):

- `eta = 0.10`: some probability remains allocated uniformly.
- `epsilon = 1e-6`: handles zero priorities.
- Uniform mode: `p_i = 1/N` (equivalent to eta=1).
- Require finite, nonnegative priorities, finite epsilon > 0, and finite eta
  satisfying 0 < eta <= 1. Validate the complete batch before consuming the RNG.
- Sum/normalize in double precision. Equal or all-zero priorities give uniform
  probabilities. Every point has probability at least eta/N.
- Use `std::discrete_distribution` and an explicitly supplied persistent
  `std::mt19937`. No custom alias table, CUDA RNG, or sampler class hierarchy.
- Return p_i from the frozen distribution used for that draw, before updating
  priorities. These are per-draw probabilities, not probabilities of appearing
  at least once in the whole minibatch.
- Account for floating-point normalization: use the categorical distribution's
  normalized probabilities when reporting draws. Check normalization and the
  exploration floor with a numerical tolerance, not exact floating-point equality.

### 3. Residual and priority updates

For each sampled point, average squared error over value channels:

```text
r[b,k] = mean_c((result.model_output[b,k,c] - result.loss_target[b,k,c])^2)
```

In MSE mode this is RGB error. In CICFM mode it is **velocity prediction error**,
using the actual sampled time/noise. Do not compare predicted velocity with RGB.
Residuals are unweighted and come from the pre-update forward pass already
returned by trainStep; do not run a second forward pass just to get them.

Use an exponential moving average to reduce noisy updates:

```text
priority[b,i] = beta * old_priority[b,i] + (1-beta) * observed_mean_residual[b,i]
beta = 0.90 by default; require finite beta with 0 <= beta < 1
```

If an original point appears multiple times, average its residuals first, then
update it once. Unobserved points keep their previous priorities. Validate the
entire update before mutating state; invalid indices, shapes, or nonfinite/negative
residuals must fail rather than poison future distributions. Check the converted
stored priority remains finite. No full-grid refresh every training step.

For the uniform baseline, leave priorities unchanged and skip priority updates;
its draw probabilities are always 1/N. Continue computing residuals for comparable
diagnostics. Priority-update atomicity applies to priorities only: trainStep has
already updated the model, so any subsequent update failure aborts the run rather
than claiming that the model and RNG were rolled back.

### 4. Objective and importance correction

Week 6 v1 deliberately uses the existing unweighted loss. Its expected coordinate
objective is `sum_i p_i * error_i`, not the uniform full-grid mean. Document this
as **uncorrected priority sampling**. It emphasizes difficult points; it is not
an unbiased estimator of the original coordinate objective.

If preserving that objective becomes a requirement, make it a separate commit
before experiments: with replacement, multiply each per-point loss AND its
output gradient by `1/(N*p_i)`, using draw-time probabilities. Reduce over B*K
points and C channels. Do not normalize by the sum of weights or deduplicate
draws; those change the estimator. Test the expectation on a tiny enumerable
example. This extension is intentionally not required for Week 6 v1.

### 5. Reproducibility and feature layout

- Keep separate random streams/settings for model initialization, coordinate
  selection, training noise/time, and fixed evaluation probes.
- Seed the coordinate RNG once, then advance it. Re-running with the same seed
  on the same toolchain reproduces the sequence; do not promise identical draws
  across different standard-library implementations.
- The existing train step reconstructs RNGs from config.seed. For training step s,
  supply a documented seed such as `noise_base + 2*s`; the existing implementation
  uses the next seed for time. Avoid seed-range wrap in the demo. Evaluation uses
  separate fixed seeds and does not consume the training streams.
- Begin integration with the deterministic CICFM probe. Then enable new random
  noise/time each step. Actual adaptive experiments need the random mode too.
- Flatten `[B,K,D]` to `[B*K,D]`, call existing fourierEmbedding, and reshape to
  `[B,K,D+2*D*F]`. Tensor reshape currently copies, so include that cost in timing.
- Use CPU training embeddings; do not substitute the current GPU embedding layout.

## Proposed files and ownership

Keep a small interface made of ordinary structs and functions:

- `include/sampling/coordinate_sampler.h`, `src/sampling/coordinate_sampler.cpp`:
  `SamplingConfig`, `SampleSelection`, probability construction, index drawing,
  and priority updates. SampleSelection carries B/N/K plus index/probability
  vectors. The caller owns priorities and the RNG; the sampler owns no model.
- `include/ops/coordinate_batch.h`, `src/ops/coordinate_batch.cpp`: add
  `gatherCoordinateBatch(batch, indices, sample_count)`. It validates exactly B*K
  indices and their ranges; it preserves requested order and duplicates. It does
  not need probabilities or know anything about loss modes.
- `include/training/sampled_train_step.h`, `src/training/sampled_train_step.cpp`:
  a small reusable composition function that selects, gathers, embeds, invokes
  existing trainStep, computes residuals, and updates priorities after a successful
  step. Return the selection, train result, and residuals. Keep this testable
  outside a command-line executable; add timing fields only if the demo needs them.
- `examples/train_adaptive_sampling.cpp`: finite, configurable training run on a
  generated image with both smooth and detailed regions. Own the model, complete
  batch, priorities, seeds, evaluation schedule, and CSV logging here.
- Add focused tests alongside those files, and wire sources/targets into CMake.
  No Tensor redesign, attention dependency, or new third-party dependency.

## Milestones and commits

Days are sequencing targets, not a promise that every milestone takes one day.
Each commit should build and have its relevant tests pass before continuing.

### Day 1 — Freeze the contract and establish a runnable baseline

Commit: `docs(sampling): define CPU adaptive sampling contract`

- Finalize this document's replacement rule, shapes, priority math, seed policy,
  uncorrected objective, and acceptance gates; retain the reviewed baseline SHA.
- Build this branch in Debug and run all eight existing CPU tests.
- Optionally register those executables with CTest in a separate small commit:
  `test: register CPU tests with CTest`. Existing tests use assert, so Debug matters.
- Gate: no uncertainty about what indices/probabilities mean, where priorities
  live, or which objective this version optimizes.

Baseline validation commands (run from the repository root):

```sh
cmake -S . -B /tmp/tinyinr-week6-commit1 -DCMAKE_BUILD_TYPE=Debug
cmake --build /tmp/tinyinr-week6-commit1 -j 4
for test_name in \
  test_fourier_embedding_tensor test_mlp_forward test_loss test_cicfm \
  test_mlp_backward test_training_debug test_train_step test_full_size_baseline
do
  /tmp/tinyinr-week6-commit1/"$test_name" || exit 1
done
```

Run each command only after the preceding command succeeds. The baseline gate is
a successful Debug build and all eight executables reporting PASS; compilation
alone or an empty CTest run is not sufficient. Existing assert-based checks must
remain enabled. CUDA validation is outside this CPU-only commit.

### Day 2 — Priorities -> probabilities -> original indices

Commit: `feat(sampling): add seeded uniform and adaptive coordinate draws`

Implemented in `include/sampling/coordinate_sampler.h` and
`src/sampling/coordinate_sampler.cpp`. `coordinateSamplingProbabilities` returns
batch-major B*N probabilities; `sampleCoordinateIndices` returns B*K original
indices and draw-time probabilities using a caller-owned RNG. SamplingConfig
uses float eta/epsilon with double-precision probability calculations. Set eta=1
for uniform draws. Priority initialization remains the caller's responsibility.

Validation: all nine Debug CPU test executables pass, including the new
`test_coordinate_sampler`. The sampler test also passes with AddressSanitizer,
UndefinedBehaviorSanitizer, and compiler warnings treated as errors. Statistical
checks use tolerances and fixed seeds rather than a vendor-specific sequence.

- Add sampling header/source and `tests/test_coordinate_sampler.cpp`.
- Implement the probability rule and return draw-time probabilities with indices.
- Add CPU library/test wiring to CMake.
- Test uniform/equal/all-zero priorities, invalid inputs, row normalization,
  exploration floor, index bounds, B>1 row independence, valid duplicate draws,
  and reproducibility with two identically seeded RNGs.
- Test a fixed known distribution over many draws with a generous tolerance;
  do not require one exact vendor-specific sequence or exact counts.
- Gate: raising a controlled point priority increases its probability and observed
  frequency, while every other point retains a nonzero chance.

### Day 3 — Gather the selected data without losing identity

Commit: `feat(coordinates): gather coordinate-value pairs by source index`

- Add gatherCoordinateBatch and `tests/test_coordinate_batch.cpp`.
- Use marker values to prove coordinates and targets stay aligned for B>1,
  reordered selections, duplicates, K=1, and K=N.
- Reject malformed lengths/out-of-range indices; verify the source batch is unchanged.
- Keep randomSubsample's current interface and no-replacement behavior intact.
- Gate: selection metadata can always map a sampled row back to its original point.

### Day 4 — Run the actual uniform sampled training path

Commit: `feat(training): connect sampled coordinates to CPU train steps`

- Add sampled_train_step header/source, initially with uniform selection and no
  adaptive updates. Add `tests/test_sampled_train_step.cpp`.
- Gather -> flatten -> CPU embedding -> reshape -> existing trainStep.
- Return selection and channel-mean squared residuals from existing train results.
- Verify B>1/F=8 shapes and a small F=16 case; explicitly test residual arithmetic
  with known output/target values and CICFM velocity rather than RGB targets.
- Compare one composed step with a manually assembled step from identical model,
  RNG, and noise/time seeds: same selection, loss, and updated parameters.
- First use MSE and deterministic CICFM; then test seeded random CICFM.
- Add scalar-loss finiteness validation before SGD and a finite-input overflow
  regression in the existing train-step tests.
- Gate: a real coordinate batch completes a learning step and reports finite
  residuals attached to the correct original points. No attention code is involved.

### Day 5 — Close the adaptive feedback loop

Commit: `feat(sampling): update point priorities from training residuals`

- Implement duplicate aggregation and one EMA update per observed original point.
- Wire updates into the sampled training composition after successful training.
- Unit tests: exact EMA arithmetic, unchanged unobserved points, B>1 isolation,
  duplicates/order independence, and rejection without partial mutation.
- Integration test: controlled feedback changes the next probability distribution;
  a fixed-seed multi-step run reproduces losses, selections, and final priorities.
- Gate: sample -> train -> residual -> priority -> next sample works in both modes.
  Do not assert that every step's loss falls or that adaptive always beats uniform.

### Day 6 — Add a runnable demo and honest measurements

Commit: `feat(examples): add reproducible adaptive sampling training demo`

- Add the CPU-only example target and options for mode, sampling strategy, N/K,
  steps, hidden width, F, learning rate, seeds, eta/beta, and CSV output path.
- Start small (for example a 16x16 RGB synthetic image, K=64, hidden width 16);
  preserve three hidden layers. Run a short separate width-256 smoke check.
- Compare uniform and adaptive with-replacement selection using identical data,
  initialization, learning rate, K, steps, and noise/time seed schedule.
- MSE evaluation: forward-only average error over every original point.
- CICFM evaluation: forward-only full-grid velocity error over a small fixed bank
  of z0/t probes, identical across runs and evaluations. Reuse makeCicfmBatch,
  mlpForward, and mseLoss; never call trainStep for evaluation because it updates
  the model. Label this velocity error, not reconstruction or generation quality.
- Log step, mode/strategy/seeds/configuration, sampled loss, fixed evaluation loss,
  per-step unique count, cumulative point coverage, duplicate rate, probability
  min/max, priority min/max, finite status, sampling/gather/embedding/train/update
  times, and total training-step time. Record evaluation time separately.
- Count coverage by original (example, point) identity. Use the same evaluation
  interval in both runs. Make no speedup claims from Debug builds.
- Gate: one command per strategy produces finite logs and comparable measurements.

### Day 7 — Acceptance evidence and handoff

Commit: `test(sampling): verify CPU adaptive loop and document results`

- Run the full existing and new Debug CPU tests. Add any missing edge regressions.
- Run short uniform/adaptive demos in both MSE and CICFM, plus a width-256 smoke run.
- If publishing timings, use the same Release build for both strategies and record
  CPU, build settings, seed, sizes, and model configuration. Keep correctness checks
  explicit in the demo even when assert is disabled.
- Save a small representative CSV and `docs/week6_adaptive_sampling_results.md`
  with commands, configuration, fixed-evaluation curves or tables, timing overhead,
  coverage, and limitations. Distinguish smoke evidence from research conclusions.
- Update `docs/milestones.md` only for completed gates.
- Gate: another person can reproduce the CPU loop from the documented commands.

## Coordination with the attention work

Agree now on the seam: sampling chooses which original points exist in a minibatch;
attention later receives their gathered coordinate features. Source indices travel
alongside those features so feedback can return to original points. Sampler
probabilities do not belong inside the attention score/softmax calculation.

Partner-owned work: attention header/source, synthetic latent fixtures, attention
CUDA kernels/tests/benchmarks. This branch does not require that work to be merged.
Use shared conventions but add CMake targets in small independent blocks to ease
merges. Avoid either person changing MLP inputs, TrainStepConfig, or CICFM input
assembly just to accommodate a future combined architecture.

The previously inspected attention branch already contains multi-head CPU
attention with fused K/V and an output projection; the pasted plan's single-head
CPU work is therefore partly stale. A one-head configuration is a sensible first
CUDA validation case. It does not require replacing the CPU reference. Its
coordinate wrapper was commented out when reviewed; partner should check its
build/test status before treating it as an oracle. Existing attention weight
layout differs from the MLP's `[out,in]` layout, so reusing CUDA linear operations
also needs an explicit layout agreement. These are partner tasks, not Week 6
sampling prerequisites.

This sampling branch has no learned latent producer. Synthetic latents keep the
attention primitive independently testable. Shared-context semantics and joint
training remain future design decisions.

## Definition of done and deferred work

Week 6 is done when seeded selections/probabilities, aligned gathers, correct
residual updates, both CPU training modes, fixed evaluation, and reproducible
logs all work, with the objective limitation documented and existing tests passing.

Deferred: CUDA sampling and CPU/GPU distribution checks, importance-corrected
training, multi-seed research experiments, optimized probability tables,
periodic full-dataset priority refresh, dataset-scale state/checkpoints, adaptive
K, attention integration, and ODE-based image generation. CUDA completion and a
positive speed/accuracy result are not Week 6 acceptance conditions.

If time runs short, keep the Day 2-5 correctness gates and reduce demo sizes and
presentation work. Never drop source identity, reproducibility, or the distinction
between sampled loss and fixed full-grid evaluation.
