# GPU adaptive sampling: commit 1 baseline

Status: local preparation for commit 1a is ready; NVIDIA execution and toolkit
compatibility for commit 1b are pending. AWS quota approval is not required.
Do not mark 1b or any GPU implementation milestone complete using local CPU tests.

Source baseline: `910f64406178b36e9c3e35a27f2bf604a5756619`.
Implementation branch: `codex/gpu-adaptive-sampling`.
Scope: freeze the contract and establish a real NVIDIA baseline before adding
sampling kernels. No production source or build-standard changes made yet.

## Access decision

The user selected Vast.ai and supplied a working SSH connection. Use GPU 0 only.
The instance reports no persistent workspace volume; export evidence locally
before recycle/destroy. Provider geography and immutable image digest are not
exposed by the inspected host metadata and are not assumed.

## Local environment and evidence

Host: Darwin arm64; Apple Clang 21.0.0 (`clang-2100.1.1.101`), CMake 4.4.0,
Python 3.15.0. Neither nvcc nor nvidia-smi is available on PATH; CMake reports
`CUDA compiler - NOTFOUND`. Local checks therefore cover CPU behavior only.

| Check | Status |
|---|---|
| Fresh Debug CPU build | PASS, exit 0 |
| Twelve CPU executables | All PASS, exit 0 |
| Adaptive demo CLI regression | PASS, exit 0 |
| Fallback provider/location, GPU and memory | Not supplied |
| NVIDIA driver, nvcc and GPU architecture | Not checked |
| CUB/cuRAND headers and compiler compatibility | Not checked |
| Seven existing CUDA executables | Not run |
| GPU baseline acceptance gate | Pending |

Commands used locally, from the repository root:

```sh
cmake -S . -B /tmp/tinyinr-gpu-baseline-debug -DCMAKE_BUILD_TYPE=Debug
cmake --build /tmp/tinyinr-gpu-baseline-debug -j4
for name in test_fourier_embedding_tensor test_attention test_mlp_forward \
  test_loss test_cicfm test_mlp_backward test_training_debug test_train_step \
  test_full_size_baseline test_coordinate_sampler test_coordinate_batch \
  test_sampled_train_step; do
  /tmp/tinyinr-gpu-baseline-debug/"$name" || exit 1
done
python3 tests/test_adaptive_sampling_demo.py \
  /tmp/tinyinr-gpu-baseline-debug/train_adaptive_sampling
```

## NVIDIA host procedure (not executed)

Connection information needed: SSH host/user or alias, local identity-file path
if applicable, and checkout directory. Never place private keys in this report.
Use a separate checkout/build directory; do not overwrite another person's work.
Confirm the source SHA above, or explicitly record a new baseline if it changes.

Record provider, location, machine/image identifier, `nvidia-smi`, `nvcc --version`, `cmake --version`, host
compiler version, available GPU memory and compute capability. Record both driver
and installed toolkit versions: the CUDA version displayed by nvidia-smi is not
the installed nvcc version. Confirm toolkit CUB/cuRAND headers and version macros
with a small compile probe using the installed compiler before choosing APIs.
Keep any required compatibility fix separate and rerun the baseline after it.

From that checkout, run each command only after the preceding one succeeds:

```sh
cmake -S . -B /tmp/tinyinr-nvidia-gpu-baseline -DCMAKE_BUILD_TYPE=Debug
cmake --build /tmp/tinyinr-nvidia-gpu-baseline -j4
for name in test_fourier_embedding_tensor test_attention test_mlp_forward \
  test_loss test_cicfm test_mlp_backward test_training_debug test_train_step \
  test_full_size_baseline test_coordinate_sampler test_coordinate_batch \
  test_sampled_train_step test_fourier_embedding_cuda test_cpu_cuda_parity \
  test_silu_cuda test_mlp_forward_cuda test_mse_cuda test_mlp_backward_cuda \
  test_train_step_cuda; do
  /tmp/tinyinr-nvidia-gpu-baseline/"$name" || exit 1
done
python3 tests/test_adaptive_sampling_demo.py \
  /tmp/tinyinr-nvidia-gpu-baseline/train_adaptive_sampling
```

Require CUDA detection, not just a successful CPU-only configure. Existing C++
tests use assert, so Debug is required; an empty CTest run is not evidence.
Record individual failures, exact commands, and exit statuses. Existing CUDA
failures are prerequisites to diagnose, not grounds to loosen tolerances.

## Frozen v1 contract for subsequent commits

The [implementation plan](gpu_adaptive_sampling_plan.md) defines probability,
replacement, residual, duplicate-averaging, and EMA semantics. These additional
constraints make the execution and validation boundary explicit.

### Model and buffer support

- Exactly three hidden layers plus one output layer, SiLU, matching the existing
  fixed-size GPU cache/model. Validate this before upload, along with model/config
  agreement. Do not accept arbitrary layer counts just because the CPU MLP does.
- Positive B/N/K/D/C/F and hidden width; K <= N. Default tests use D=2, C=3,
  F=8, with F=16 and non-RGB shapes in focused tests.
- Feature width E=D+2*D*F; input width is E for MSE, E+C+1 for CICFM. Training
  rows R=B*K. Full evaluation has R_eval=B*N and needs independent size checks.
- Check arithmetic before multiplication/addition or allocation. Dimensions,
  flattened offsets and launch counts used by existing int-indexed kernels must
  fit signed int. Account for block-rounding additions in compute_grid_size and
  tiled launches, not merely element counts. Allocation bytes must fit size_t;
  reject unsupported requests before partial allocation or launch.
- Validate products for full coordinates/targets, features, model weights,
  activations and gradients, not just R. Use int64_t point metadata; check any
  narrowing explicitly. Composite sort keys use checked uint64_t arithmetic.
- Match the CPU guard 0 < F < float max_exponent; additionally reject any
  nonfinite embedding phase/output. This guard alone does not guarantee finite
  trigonometric arguments for arbitrary coordinates or large F.
- Float model/data/priorities/residuals; double distribution/CDF and grouped sums.
  Retain finite/nonnegative priority validation even when eta=1. Reject a positive
  probability whose CDF interval collapses numerically rather than hiding it.
- GPU memory exhaustion is an explicit error, not a silently reduced N or K.
  Actual practical limits and toolkit restrictions remain to be measured on the selected NVIDIA host.

### Numerical comparison and repeatability

All numerical comparisons explicitly reject NaN/Inf before testing a tolerance.
Use `abs(actual-reference) <= abs_tol + rel_tol*abs(reference)`.

| Quantity | Initial acceptance rule |
|---|---|
| Gathered RGB targets; supplied selection indices | Exact equality |
| Shape, original-point identity, duplicate counts | Exact equality |
| Probabilities, CDF row totals, exploration floor | Absolute tolerance 1e-12 plus relative tolerance 1e-10; also require monotonic CDF and positive representable intervals |
| Fourier features at F=8/F=16 | abs_tol=1e-5, rel_tol=1e-4; verify exact positions of raw coordinates separately |
| Residual/EMA from identical supplied outputs and targets | abs_tol=1e-6, rel_tol=1e-5 |
| Existing five-step CPU/GPU training loss | abs_tol=1e-6, rel_tol=1e-3, matching test_train_step_cuda |
| Existing five-step model parameters | abs_tol=1e-5, rel_tol=1e-3, matching test_train_step_cuda |

These are test acceptance targets, not GPU measurements. Investigate any failure
before changing them; report a justified change explicitly. For integrated tests,
compare residual arithmetic against the downloaded GPU prediction/target as well
as comparing GPU predictions with CPU predictions, so training drift is not confused
with a residual-kernel bug. Do not require independent adaptive trajectories to
choose identical points after floating-point differences have changed priorities.

For fixed priorities and fixed seed/step, GPU selection must replay on the same
toolchain/device. For supplied sorted observations, accumulation order is fixed.
Do not require CPU std::discrete_distribution to match GPU Philox draws. Test
injected uniform values to isolate CDF search from RNG; independently test
frequencies over many draws with a predeclared statistical tolerance. Replaying
the whole trained model is a separate check, not guaranteed by RNG replay alone.
Point selection and CICFM noise/time use separate random sequences. Evaluation
must not consume or reset these sequences.

### Ownership, failure and synchronization

Allocate persistent buffers and reusable scratch once. Device input pointers are
borrowed; freeing a training context must not free its caller's data. Use the
default stream until all training stages support an explicit stream consistently.
Validation kernels must finish before status is consumed. A block-local barrier
is not a grid-wide validation barrier. Invalid input must gate downstream gathers
and training; finite checks precede SGD. Candidate priority writes are published
only if all validation succeeds. On later failures abort the run; model/RNG
rollback is not promised. Small status readbacks are permitted and measured.

Keep the existing unweighted objective, with replacement, and retain the original
CPU path. CUDA language-standard changes, if required by the selected toolkit, must be
targeted and validated rather than guessed locally.
