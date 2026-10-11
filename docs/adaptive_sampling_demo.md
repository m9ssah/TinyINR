# CPU adaptive sampling demo

The demo trains a three-hidden-layer SiLU MLP on a generated RGB image. Its red
and green channels are smooth gradients; its blue channel is constant on the left
and a checkerboard on the right. The default 256 points form a 16x16 image.
Other point counts use a nearly square grid, with an incomplete last row if needed.

## Build and compare

Run from the repository root:

```sh
cmake -S . -B /tmp/tinyinr-demo -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/tinyinr-demo --target train_adaptive_sampling -j 4
for mode in mse cicfm; do
  for strategy in uniform adaptive; do
    /tmp/tinyinr-demo/train_adaptive_sampling --mode "$mode" \
      --sampling "$strategy" --csv "/tmp/$mode-$strategy.csv" || exit 1
  done
done
```

Each command overwrites its CSV path. Both strategies draw with replacement, so
a minibatch can contain duplicate points. Keep all settings except strategy and
output path identical when comparing strategies within a mode. Adaptive sampling
uses an unweighted loss: it changes which points receive more training weight.
It is not an unbiased estimate of uniform full-grid loss, and improvement over
uniform sampling is not guaranteed.

## Options

All options take one value. `--help` prints the defaults.

| Option | Default | Meaning |
|---|---|---|
| `--mode` | `cicfm` | `mse` predicts RGB; `cicfm` predicts velocity. |
| `--sampling` | `adaptive` | `uniform` or `adaptive`. |
| `--points`, `--samples` | `256`, `64` | Original points N and draws K per step; 1 <= K <= N. |
| `--steps`, `--hidden`, `--frequencies` | `20`, `16`, `8` | Training steps, hidden width, Fourier frequencies. |
| `--learning-rate` | `0.001` | Positive SGD learning rate. |
| `--eta`, `--beta` | `0.10`, `0.90` | Uniform exploration fraction and priority EMA retention. Uniform mode forces eta=1 and skips priority updates. |
| `--model-seed`, `--sampling-seed` | `17`, `42` | Model initialization and persistent coordinate RNG seeds. |
| `--noise-seed`, `--eval-seed` | `123`, `100000` | Separate training and evaluation noise/time seed bases. |
| `--eval-every`, `--eval-probes` | `5`, `3` | Evaluation interval and fixed CICFM probe-bank size. MSE uses one full-grid evaluation. |
| `--csv` | `adaptive_sampling.csv` | Output path; its parent directory must exist. |

The sampler's epsilon remains fixed at `1e-6`. Batch size is one, coordinate
dimension is two, RGB channels are three, and there are three hidden layers.
Training step 1 uses the noise base; each subsequent step adds two. Time uses
the noise seed plus one. Evaluation probe seeds follow the same pattern from
their separate base. Seed schedules that would wrap are rejected. Repeated
settings reproduce non-timing results on the same toolchain; standard-library
random distributions need not match across platforms.

## Reading the CSV

- Step 0 evaluates the initial model. Its sampled loss and training seed are blank.
- `sampled_loss` describes the minibatch before that step's SGD update.
- `eval_loss` describes the model after the update, using every original point.
  It is recorded at step 0, every evaluation interval, and the final step; other
  rows leave it blank. MSE reports `rgb_mse`. CICFM reports `velocity_mse`, averaged
  over the same fixed noise/time probes each time. This is not image reconstruction
  quality. Evaluation only calls forward/loss functions and never trains the model.
- `unique_points` counts distinct points in that step; `coverage_count` counts
  distinct original points seen so far. `duplicate_rate` is `1 - unique_points/K`.
- Probability extrema describe the distribution before sampling. Priority extrema
  describe priorities after the update. Uniform priorities remain one.
- `all_finite=1` is written after the run's finite checks succeed; numerical or
  file errors stop the executable with a nonzero exit status.
- Stage times are milliseconds inside the sampled training function, including
  embedding reshapes/copies. `train_ms` includes forward, loss, backward, and SGD;
  `residual_ms` and `update_ms` measure feedback work. `step_ms` includes validation
  and these stages, but excludes caller diagnostics, CSV I/O, evaluation, setup,
  and return-time cleanup. It is not total program runtime. `eval_ms` measures
  forward/loss evaluation separately; preparing the fixed probes is setup work.

Build type is included in every row. Use matching Release builds for timing
comparisons; these short runs establish functionality, not speedup evidence.

## Checks

The CLI regression script needs only Python 3's standard library:

```sh
python3 tests/test_adaptive_sampling_demo.py /tmp/tinyinr-demo/train_adaptive_sampling
/tmp/tinyinr-demo/train_adaptive_sampling --hidden 256 --points 4 --samples 2 \
  --steps 1 --eval-probes 2 --csv /tmp/width256-smoke.csv
```

The script checks both modes and strategies, CSV contents, seeded replay,
evaluation independence, and rejected arguments. The width-256 command is a
small architecture smoke test. Run the existing C++ tests in Debug so their
assertions remain enabled. No CUDA or attention implementation is required.
