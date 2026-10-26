# Week 6 CPU adaptive sampling acceptance results

The CPU sampling loop passes its Week 6 acceptance checks: seeded draws, aligned
coordinate/value gathering, residual feedback, both training modes, fixed
evaluation, and reproducible CSV logging. The runs below are smoke evidence, not
a research result or a claim that adaptive sampling is faster or more accurate.

Production code tested: `d48fe7f` (commit 6). Commit 7 adds CLI edge regressions,
these saved results, and completion documentation; it changes no training code.
See the [contract](week6_adaptive_sampling_plan.md) and
[demo guide](adaptive_sampling_demo.md) for the algorithm and CSV definitions.

## Environment and configuration

- CPU: Apple M5, arm64; macOS 26.4 (25E246).
- Compiler: Apple Clang 21.0.0 (`clang-2100.1.1.101`); CMake 4.4.0;
  Python 3.14.5 for CLI checks.
- Correctness build: Debug, assertions enabled. Measurement build: Release,
  `-O3 -DNDEBUG -std=gnu++14`, same executable for both strategies.
- Main runs: one generated 16x16 RGB image, B=1, N=256, K=64, 20 SGD steps,
  three hidden layers of width 16, SiLU, F=8, learning rate 0.001.
- Adaptive eta=0.10, epsilon=1e-6, beta=0.90, priorities initially one.
  Uniform uses eta=1 and leaves priorities unchanged. Both draw with replacement.
- Seeds: model=17, sampling=42, training noise base=123, evaluation base=100000.
  Step s (starting at 1) uses noise seed `123 + 2*(s-1)` and time seed one higher.
- Evaluation at steps 0, 5, 10, 15, 20. MSE evaluates all RGB targets. CICFM
  evaluates full-grid velocity error over three fixed noise/time probes. These
  evaluations use forward/loss only and do not consume training RNG state.

## Reproduce validation

Run from the repository root; stop if any command fails. CTest is not configured,
so run the executables explicitly. Release builds disable the C++ tests' asserts.

```sh
cmake -S . -B /tmp/tinyinr-week6-commit7-debug -DCMAKE_BUILD_TYPE=Debug
cmake --build /tmp/tinyinr-week6-commit7-debug -j 4
for name in test_fourier_embedding_tensor test_mlp_forward test_loss test_cicfm \
  test_mlp_backward test_training_debug test_train_step test_full_size_baseline \
  test_coordinate_sampler test_coordinate_batch test_sampled_train_step; do
  /tmp/tinyinr-week6-commit7-debug/"$name" || exit 1
done
python3 tests/test_adaptive_sampling_demo.py \
  /tmp/tinyinr-week6-commit7-debug/train_adaptive_sampling
```

All eleven CPU executables and the CLI script passed. Existing tests cover
probability normalization/exploration, empirical draw frequencies, seeded replay,
B>1 isolation, duplicates, source alignment, exact EMA arithmetic, invalid updates
without partial writes, known RGB/velocity residuals, and multi-step feedback.

The CLI checks passed in Debug and Release, and against the commit-6 sanitizer
build (AddressSanitizer + UndefinedBehaviorSanitizer). Added edge checks cover
N=K=1, N=K=7 (incomplete image grid), beta=0 with eta=1, the highest valid noise/time
seed pair, final evaluation before the scheduled interval, and eta=1 parity with
uniform sampling in both modes. Normal adaptive runs must retain their exploration
floor, nonnegative priorities, and a changed next-step distribution. Existing CLI
checks cover replay, evaluation independence, malformed arguments, and CSV errors.

## Reproduce saved runs

```sh
cmake -S . -B /tmp/tinyinr-week6-commit7-release -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/tinyinr-week6-commit7-release --target train_adaptive_sampling -j 4
python3 tests/test_adaptive_sampling_demo.py \
  /tmp/tinyinr-week6-commit7-release/train_adaptive_sampling
mkdir -p /tmp/tinyinr-week6-commit7-results
for mode in mse cicfm; do
  for strategy in uniform adaptive; do
    /tmp/tinyinr-week6-commit7-release/train_adaptive_sampling \
      --mode "$mode" --sampling "$strategy" --steps 20 \
      --csv "/tmp/tinyinr-week6-commit7-results/$mode-$strategy.csv" || exit 1
  done
  /tmp/tinyinr-week6-commit7-release/train_adaptive_sampling \
    --mode "$mode" --hidden 256 --points 4 --samples 2 --steps 1 --eval-probes 2 \
    --csv "/tmp/tinyinr-week6-commit7-results/$mode-width256.csv" || exit 1
done
```

All omitted options use the defaults listed above. Outputs overwrite those paths.
The six original CSVs are saved under
[`benchmarks/results/week6_adaptive_sampling`](../benchmarks/results/week6_adaptive_sampling/).
No timing values were edited or filtered. Repeated runs on the same toolchain
should reproduce non-timing columns; timing columns vary.

## Fixed evaluation and coverage

Each entry is the mean squared error, rounded to six decimal places. RGB and
velocity errors measure different targets and should not be compared directly.

| Step | MSE uniform | MSE adaptive | CICFM uniform | CICFM adaptive |
|---|---:|---:|---:|---:|
| 0 | 0.818098 | 0.818098 | 0.805835 | 0.805835 |
| 5 | 0.790360 | 0.787098 | 0.756460 | 0.751227 |
| 10 | 0.766174 | 0.757253 | 0.711797 | 0.693486 |
| 15 | 0.743532 | 0.730467 | 0.673019 | 0.646779 |
| 20 | 0.722598 | 0.699868 | 0.640050 | 0.603040 |

| Run | Distinct points seen / 256 | Mean per-step duplicate fraction |
|---|---:|---:|
| MSE uniform | 253 | 0.108594 |
| MSE adaptive | 254 | 0.122656 |
| CICFM uniform | 253 | 0.108594 |
| CICFM adaptive | 255 | 0.119531 |

Adaptive had lower final fixed evaluation error in this seed's short runs. There
are no repeated-seed estimates or uncertainty bounds. Uniform coverage matches
between modes because its coordinate RNG and fixed probabilities are identical.
Every saved row reports finite diagnostics. Width-256 adaptive smoke checks
(N=4, K=2, one step) also passed: MSE RGB error moved from 0.887380 to 0.742855;
CICFM velocity error moved from 0.390408 to 0.373853. These tiny checks establish
that the wider architecture runs; they do not establish convergence.

## Timing and feedback overhead

Arithmetic means in milliseconds over the 20 training rows, excluding step zero:

| Run | Sampling | Gather | Embedding | Train | Residual | Update | Total step |
|---|---:|---:|---:|---:|---:|---:|---:|
| MSE uniform | 0.003938 | 0.016506 | 0.014506 | 7.776327 | 0.010810 | 0 | 7.822258 |
| MSE adaptive | 0.002540 | 0.014221 | 0.012579 | 7.165673 | 0.008646 | 0.001646 | 7.205396 |
| CICFM uniform | 0.003277 | 0.013496 | 0.013287 | 7.827383 | 0.009469 | 0 | 7.867010 |
| CICFM adaptive | 0.002485 | 0.012727 | 0.012706 | 7.681898 | 0.009010 | 0.001512 | 7.720442 |

The measured priority-update stage adds about 0.0015–0.0016 ms per adaptive step.
Residual computation also costs time and is included for both strategies. This
breakdown measures stages; subtracting whole-run times would not isolate the
adaptive overhead. Runs were sequential, without controlled warmup, repetition,
or machine-load isolation. The lower adaptive total times are not speedup evidence.

Mean evaluation time across five evaluations was 11.628867 ms (MSE uniform),
9.611650 ms (MSE adaptive), 30.465833 ms (CICFM uniform), and 30.456833 ms (CICFM
adaptive). Evaluation is excluded from step times. Setup, fixed-probe preparation,
caller diagnostics, CSV output, and return-time cleanup are also excluded; total
step is not whole-program runtime. Small differences between stage sums and total
include validation and instrumentation overhead.

## Limits and handoff

Week 6 uses **uncorrected priority sampling**: the expected training objective is
weighted by draw probabilities. It is not an unbiased uniform-grid loss estimator.
CICFM evaluation measures velocity error, not ODE-generated image quality.

CUDA was unavailable and was not tested. CUDA sampling, attention integration,
importance correction, multi-image identity/checkpointing, and multi-seed quality
or speed experiments remain future work. The integration boundary is unchanged:
sampling returns original point indices and draw-time probabilities; gathering
keeps coordinates/targets aligned; residual feedback updates original priorities.
Probabilities do not belong inside attention's softmax.
