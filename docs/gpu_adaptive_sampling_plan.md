# GPU adaptive sampling implementation plan

Status: commits 1a and 1b complete on `codex/gpu-adaptive-sampling`; local CPU
and Vast NVIDIA baseline checks passed. No GPU sampling code implemented yet.
See the [baseline record](gpu_adaptive_sampling_baseline.md) for executed checks,
exported logs and the installed CUB requirement for C++17. Next: commit 2.
Baseline inspected: `910f64406178b36e9c3e35a27f2bf604a5756619` (includes CPU
sampling, attention build fix, and CPU embedding-limit validation updates).

## Goal and scope

Move the existing CPU adaptive loop onto an NVIDIA GPU while retaining its
meaning: sample with replacement, gather aligned data, train, compute residuals,
and update original-point priorities. Support uniform/adaptive and MSE/CICFM.
Keep full data, priorities, sampled indices, features, and model on device between
steps. The CPU implementation remains the reference and remains usable without CUDA.

Use a separate branch such as `codex/gpu-adaptive-sampling` and small reviewable
commits. This is the next implementation phase, not a promise to finish all GPU
work in one calendar week. Complete each correctness gate before optimizing.
A CUDA-enabled NVIDIA machine is required for runtime acceptance. The user has
provided a working Vast.ai instance; AWS quota approval is no longer relevant.
Develop and run CPU checks on the Mac; build and execute CUDA tests on Vast.
Use only GPU 0 (`CUDA_VISIBLE_DEVICES=0`) for the baseline and comparisons.

## GPU environment and storage

The selected host provides two RTX 4060 Ti GPUs with 16 GB each. Verify the
installed toolkit separately from driver support, and record the exact source SHA
and environment in the baseline report. Use isolated source/build directories.
The host reports that its workspace is not backed by a persistent volume: export
logs and results to the Mac before recycling or destroying the instance.
No instance lifecycle changes or new rentals are part of this implementation.

## What the repository already provides

| Existing code | Reuse or required change |
|---|---|
| `src/sampling/coordinate_sampler.cpp` | Reference probabilities, selection metadata, duplicate-averaged EMA updates. |
| `src/training/sampled_train_step.cpp` | Reference orchestration and per-point residual semantics. |
| `include/kernel/train_step.cuh`, `src/kernels/train_step.cu` | Existing GPU MSE/CICFM training; currently receives host Tensors and uploads inputs every call. Add a device-input path and retain the host wrapper. |
| `include/kernel/gpu_mlp.cuh` | Existing model/cache allocation. GPU model assumes exactly three hidden layers plus an output layer. Validate that supported architecture explicitly. |
| `src/kernels/coordinate_embedding.cu` | Existing benchmark embedding omits raw coordinates and uses dimension-major ordering. It cannot directly replace CPU training embedding. |
| `src/ops/fourier_embedding_raw.cpp` | Training layout oracle: raw coordinates, then frequency-major sine/cosine pairs over dimensions; width D+2*D*F. |
| `src/kernels/cicfm.cu` | Reuse noise/target interpolation and velocity/input assembly. |
| `src/kernels/{mlp_forward,mlp_backward,mse,sgd}.cu` | Reuse learning kernels; do not reimplement the MLP. |
| `tests/test_train_step_cuda.cu` | Existing deterministic CICFM/MSE parity test; extend device/host path coverage. |
| `CMakeLists.txt` | CPU-only builds already supported. Register new CUDA files and targets inside the CUDA condition. |

## Contract to preserve

- B examples, N original points each, K draws per example, D coordinates and C
  target channels. Require 1 <= K <= N. Initial priorities are one.
- Coordinates/targets/priorities: `[B,N,D]`, `[B,N,C]`, `[B,N,1]`.
  Selected local indices/probabilities: `[B,K]`. Residuals: `[B,K,1]`.
  Flatten only for existing GPU training: rows = B*K.
- `p_i = eta/N + (1-eta)*(priority_i+epsilon)/sum_j(priority_j+epsilon)`
  within each example; defaults eta=.10, epsilon=1e-6, beta=.90.
- With replacement, preserving selection order and duplicates. Record probabilities
  before priority updates. Keep integer indices integer, never stored as floats.
- Residual = mean squared error across channels, using the forward output before
  SGD and the actual loss target. CICFM target is velocity, not RGB.
- Average duplicate observations, then update each observed original point once:
  `new = beta*old + (1-beta)*mean_residual`. Leave unobserved points unchanged.
  Eta=1 skips priority updates. Loss remains unweighted/uncorrected.
- Invalid dimensions, parameter values, sizes, indices and nonfinite data must fail
  explicitly. Validate host metadata before allocation/launch. Device validation
  sets a status flag and gates dependent kernels; do not launch unchecked gathers.
  Priority updates use a candidate buffer and publish only after complete validation.
  Model/RNG rollback after runtime failure is not promised; abort the failed run.
- Own buffers in a small GPU sampling context, allocated once, released once.
  Retain checked sizes for all allocations. Existing int-indexed kernels require
  explicit limits on dimensions/products; do not silently narrow int64_t values.
- Start on the existing default CUDA stream. Do not add a partial multi-stream API
  while downstream MLP kernels still launch on the default stream.

## Planned algorithm

Frozen v1 choices: preallocate device buffers; use the existing default stream;
reuse GPU training and toolkit scan/sort primitives; fuse gather with training
embedding; fuse each duplicate group's averaging with its EMA candidate update.
Keep sampling, training and residual computation separate. Use sorted duplicate
aggregation rather than floating-point atomic sums. Uniform selection has a direct
integer draw path and returns 1/N without building a CDF. Adaptive selection uses
the CDF path below. Keep complete validation in both paths.

Use a cumulative distribution (CDF) and binary search first. Build probabilities
and an inclusive prefix sum separately for each example. One draw thread generates
one random number and finds its point in that example's CDF. Complexity is O(B*N)
for distribution work plus O(B*K*log N) for selection, excluding training.

Use double precision for probability sums/CDF and duplicate residual accumulation,
with float stored priorities and residuals as in the CPU reference. Normalize the
CDF by its final total; return the corresponding normalized draw-time probability.
Define search as the first CDF entry strictly greater than a uniform number in
[0,1). Test exact boundaries, first/last bins, N=1 and highly unequal priorities.
Detect unrepresentable/collapsed positive intervals at extreme configurations;
do not silently remove a point's exploration probability or blindly clamp an
out-of-range draw to the last bin. Document any narrower supported numerical range.

Use toolkit-provided CUB reduction/scan/sort primitives rather than implementing a
new general-purpose scan library. A per-example scan launch is sufficient initially;
reuse scratch buffers. Check installed toolkit APIs and language requirements first:
the Vast baseline confirms installed CUB 3.3.4 requires C++17, while existing
code builds with C++/CUDA 14. Limit any
required language-standard change to affected CUDA targets and test existing targets.

Use cuRAND Philox for GPU random generation, with distinct streams/seeds for point
selection, training noise, and training time. Document a collision-free mapping
from seed, step and draw/element index to random sequence position, with checked
counter limits. Handle cuRAND's (0,1] uniform convention explicitly when converting
to the sampler's [0,1) convention. Same seed is not expected to reproduce CPU
std::discrete_distribution indices. Reproducibility is scoped to the same GPU
implementation/toolchain, not cross-device or cross-library bitwise equality.

For duplicate updates, first sort residual observations by original point and
original draw order, then sum each group in that fixed order and write one candidate
priority. A checked composite integer key can encode both fields; reject overflow.
Reuse preallocated sort buffers. This avoids floating-point atomic additions whose
ordering could perturb future adaptive draws. One thread per group with a serial
sum is a reasonable first version; benchmark the all-duplicate case before optimizing.
Do not promise bitwise repeatability of the existing entire GPU training path:
its loss reduction and arithmetic also need separate consideration.

## Commits and gates

### 1a. `docs(cuda): freeze adaptive sampling contract and local baseline`

Commit the plan and `docs/gpu_adaptive_sampling_baseline.md`, including the tested
source SHA, local environment, CPU results, supported shapes, numerical tolerances,
replay expectations and exact remote commands. Mark every unrun GPU check pending.
This preparatory commit can be completed without AWS or any GPU rental.

Gate: all twelve existing CPU tests and the CLI regression pass locally; document
links/test lists are checked. This is preparation, not a passed CUDA baseline.

### 1b. `test(cuda): record NVIDIA baseline and toolkit compatibility`

On the selected NVIDIA host, record provider/location, GPU model and memory, driver,
nvcc, compiler, CMake, compute capability, image identifier, CUB/cuRAND versions and
source SHA. The CUDA version shown by nvidia-smi describes driver support; record
nvcc separately. Run from a separate checkout/build directory. Do not overwrite
another user's workspace. Pin the environment after the baseline works.

Run the twelve CPU tests plus all seven existing CUDA tests and the CLI regression.
Check CUB/cuRAND development headers using a small compiler probe. Document existing
failures separately and fix only prerequisites in separate small commits. Confirm
language-standard compatibility before changing CMake. Export command logs and
update the baseline record with actual results and any remaining tool restrictions.

Gate: an actual NVIDIA environment builds and runs the baseline. No dependency
on AWS approval. A CPU-only configure, compile-only result, unavailable GPU, or
missing CUDA test executable does not pass this gate.

Local interface/test-fixture drafting for commit 2 may proceed while arranging
access, but do not merge/mark GPU implementation milestones complete or accumulate
multiple unexecuted kernel commits. Pass 1b before accepting commit 2.

### 2. `refactor(cuda): accept device buffers in GPU training`

Update `include/kernel/train_step.cuh` and `src/kernels/train_step.cu`. Extract a
shared device-input execution path; retain `gpuTrainStep` as the uploading wrapper.
Use borrowed input pointers without transferring ownership; never free caller buffers.
Avoid per-step input H2D uploads and compulsory loss D2H in the new device path.
Expose/read the existing pre-SGD cached output and actual loss-target buffer for
residuals before either is overwritten on the next step. Add runtime validation
for mode, pointers, dimensions, model compatibility and learning rate, including
Release builds. Add finite checks before SGD and status propagation for this path.

Tests: original host wrapper versus device-input path, MSE and deterministic CICFM,
loss/output/parameters and allocation ownership. Preserve old callers and benchmark.
Gate: GPU training consumes already resident inputs with no data round-trip.

### 3. `feat(cuda): fuse sampled gathering with training embedding`

Add `include/kernel/coordinate_gather.cuh`, `src/kernels/coordinate_gather.cu`,
`tests/test_coordinate_gather_cuda.cu`. A fused gather/embedding kernel reads the
selected coordinates directly from full-data storage and writes training-layout
features plus matching targets. Do not allocate a sampled-coordinate intermediate
or add a second standalone embedding kernel. Preserve the existing embedding
kernel and its benchmarks. Start with one thread per sampled point, looping over
its D/F/C values; profile memory access before changing the thread layout. Validate
indices before dereferencing them. The CPU gather followed by Fourier embedding
is the independent reference for this fused operation.

Tests: supplied identical CPU/GPU indices, B>1, duplicates, reordering, K=1/K=N,
D/C other than 2/3, odd sizes and blocks with inactive threads. Compare embedding
against `fourierEmbedding`, including F=8/F=16 and raw-coordinate positions.
Gate: gathered values match exactly; embedding layout matches with numerical tolerance.

### 4. `feat(cuda): draw seeded uniform and adaptive coordinates`

Add `include/kernel/coordinate_sampler.cuh`, `src/kernels/coordinate_sampler.cu`,
`tests/test_coordinate_sampler_cuda.cu`. Introduce the reusable sampling context,
full-data/priorities storage and reduction/scan/RNG scratch space. Implement
validation, per-example probabilities/CDF, and binary-search sampling. For eta=1,
draw uniform integer indices directly and store probability 1/N, skipping the CDF.
Use unbiased bounded integer generation (rejection of the excess integer range),
not raw modulo. Make eta=1 take the same path regardless of the strategy label.

Tests: CPU probability parity, sum/exploration checks, invalid later batch rows,
zero/equal/skewed priorities, N=1, B>1, non-power-of-two and multi-block N, draw
boundaries, valid indices and stored probabilities, seed replay and distribution
frequencies. Allow injected uniform numbers in the lower-level draw test to separate
CDF/search correctness from RNG behavior. Do not compare CPU/GPU random indices
merely because their seeds match.
Gate: distribution and selection are correct before connecting model feedback.

### 5. `feat(cuda): update priorities from sampled residuals`

Add `include/kernel/coordinate_priority.cuh`, `src/kernels/coordinate_priority.cu`,
`tests/test_coordinate_priority_cuda.cu`. Compute channel-mean squared residuals,
validate them, sort/group duplicate indices and publish validated EMA updates.
Fuse group averaging and EMA candidate calculation in one pass after sorting.
Preserve original draw order within a group. Use preallocated candidate/scratch
buffers; uniform mode bypasses updates. Keep residual calculation separate from
the existing MSE loss kernel so scalar-loss behavior stays unchanged.

Tests: known RGB and velocity errors, nonzero CICFM noise, beta=0/default beta,
unobserved points, all draws hitting one point, different points sharing a local
index across examples, invalid final observation, overflow/nonfinite values, and
unchanged priorities after rejection. Compare duplicate results against the CPU
reference and test repeated update runs on the same device.
Gate: one bad observation cannot partially update priorities; duplicate points
receive one averaged update rather than repeated EMA updates.

### 6. `feat(cuda): connect adaptive sampling to resident GPU training`

Add `include/kernel/sampled_train_step.cuh`, `src/kernels/sampled_train_step.cu`,
`tests/test_sampled_train_step_cuda.cu`. Compose the validated stages, flattening
B*K only at the training interface. Start with MSE, then deterministic CICFM, then
random CICFM. Generate training z0/t on GPU using separate Philox streams (a small
helper kernel alongside existing CICFM code); keep externally supplied probes for
parity tests. Errors stop the loop; never continue with invalid/stale buffers.

Parity uses supplied identical indices and noise/time on CPU/GPU, comparing input
assembly, pre-update predictions, loss, residuals, new priorities, and parameters.
For random CICFM, manually assemble CPU makeCicfmBatch + forward/loss/backward/SGD
with the supplied noise; do not ask the CPU seeded generator to match Philox.
Use absolute+relative tolerances justified by existing training parity, and reject
NaN explicitly. Then run autonomous multi-step GPU feedback tests with both modes,
B>1, partial blocks, uniform unchanged priorities, and advancing random streams.
Gate: complete device-resident loop, no per-step point-array transfers or allocations;
small status/metric readbacks allowed and accounted for separately.

### 7. `feat(examples): add reproducible GPU adaptive sampling demo`

Add `examples/train_adaptive_sampling_cuda.cu` and its CUDA-only CMake target.
Match CPU demo settings/data and CSV meanings. Share small data-generation/options
helpers only where that avoids actual duplication; do not build a generic framework.
Use the same initial CPU model for CPU/GPU comparison, upload once. Allocate a
separate full-grid evaluation cache so evaluation cannot overwrite training state.
MSE evaluation is full-grid RGB error; CICFM uses a fixed, separate z0/t probe bank.
All evaluation is forward-only. Reuse GPU inference/loss kernels and download only
summary metrics; full model/data downloads belong in debug parity tests.

Log CPU-compatible diagnostics plus distribution, RNG/draw, fused gather/embedding,
training, residual, grouping/update and evaluation GPU-event timings. Report total
host wall time including waits/transfers separately. Test evaluation cadence leaves
training unchanged, invalid CLI arguments, finite output, and both strategies/modes.
Gate: one command produces a finite run with documented, reproducible settings.

### 8. `test(cuda): validate and benchmark adaptive sampling pipeline`

Run all CPU/CUDA tests in an assertions-enabled build; run Compute Sanitizer memcheck,
racecheck and synccheck on new CUDA tests on the NVIDIA host. Test explicit runtime
validation again in Release. Cover small/odd/large dimensions and width-256 smoke.
Keep tests meaningful on Release: do not rely exclusively on assert.

Add `benchmarks/bench_adaptive_sampling.cu`, saved CSVs and
`docs/gpu_adaptive_sampling_results.md`. Preallocate, warm up, repeat measured runs,
and report median/spread with hardware/build/settings. Start with N=256/4096/65536,
K=64/256/1024 (K<=N), B=1 and B>1, small and width-256 models as memory permits.
Record memory footprint, including CDF, RNG, sorting and full-grid evaluation buffers.
Compare CPU vs GPU end-to-end and GPU uniform vs GPU adaptive at matched budgets;
separate initial upload, steady-state training, evaluation, and logging costs.
Use several seeds for learning-quality claims. Draw sequences need not match across
CPU/GPU, so kernel parity and statistical quality experiments are separate evidence.

Gate: correctness evidence and honest timings. A GPU slowdown on tiny inputs is a
valid finding, not a reason to hide overhead or weaken the tests. Optimize only a
measured bottleneck in a subsequent commit.

## Proposed execution order and handoff

First deliver 1a locally, then 1b on the fallback GPU, followed by commits 2-3
(device-data plumbing). Then 4-5 (sampling
and feedback kernels). Then 6-8 (training integration, demo, acceptance evidence).
Every implementation commit includes its CMake wiring and focused tests; avoid a
final commit that is the first time any kernel is executed.

Attention remains independent: retain selected original indices beside gathered
features for later integration. This work does not invent learned latents, change
attention softmax, or add an attention backward pass. Also deferred: importance
correction, alias-table sampling, further kernel fusion, multi-stream execution,
multi-GPU training, dataset checkpointing, and custom scan/sort replacements.

## References for implementation choices

- [NVIDIA CCCL](https://github.com/NVIDIA/cccl): toolkit-provided CUB primitives and
  compiler/language compatibility; verify the installed version before choosing APIs.
- [CUB DeviceScan](https://nvidia.github.io/cccl/unstable/cub/api/structcub_1_1DeviceScan.html):
  inclusive scans; use documentation matching the installed toolkit, not unstable
  overloads merely because they appear in the latest reference.
- [cuRAND device API](https://docs.nvidia.com/cuda/curand/device-api-overview.html):
  device RNG setup and uniform endpoint conventions.
