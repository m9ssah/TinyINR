# Day 7: Final Model/Loss Design and Coding Plan

This document closes the model/loss design phase for the first trainable
TinyINR baseline. It consolidates the Day 1 through Day 6 contracts and turns
them into an implementation plan for the coding phase.

Source contracts:

- `docs/model_loss_contract.md`
- `docs/mlp_forward_mse_semantics.md`
- `docs/cicfm_scaffolding.md`
- `docs/mse_backward_verification.md`
- `docs/mse_to_cicfm_conversion.md`
- `docs/loss_transition_debugging.md`

## Final Design Summary

TinyINR will first train a coordinate-to-value INR with MSE. After the MSE path
can overfit a tiny fixed batch and pass backward checks, the same MLP family
will be rebuilt for CICFM input and trained to predict value-space velocity.

The final sequence is:

```text
coordinates -> Fourier features -> MSE MLP -> predicted values
coordinates -> Fourier features -> CICFM assembly -> CICFM MLP -> predicted velocity
```

MSE mode is the correctness anchor. CICFM mode is the controlled conversion.

## Frozen Tensor Contract

Use these names consistently in code, tests, logs, docs, and benchmark output:

| Name | Meaning | Shape |
| --- | --- | --- |
| `coordinates` | Normalized coordinate samples | `[B, N, D]` |
| `targets` | Ground-truth values | `[B, N, C]` |
| `features` | Raw coordinates plus Fourier features | `[B, N, D + 2 * D * F]` |
| `prediction` | MSE-mode model output | `[B, N, C]` |
| `z0` | CICFM source/noise value | `[B, N, C]` |
| `z1` | CICFM target/data value, equal to `targets` | `[B, N, C]` |
| `t` | CICFM time sample | `[B, N, 1]` |
| `zt` | Interpolated value | `[B, N, C]` |
| `target_velocity` | CICFM loss target | `[B, N, C]` |
| `predicted_velocity` | CICFM model output | `[B, N, C]` |

Baseline constants:

```text
D = 2
C = 3
F = 8 first, then F = 16
hidden_width = 256
linear_layer_count = 4
hidden_activation = SiLU
output_activation = identity
```

## Final Architecture

Use one MLP implementation for both modes. Only the input dimension changes.

```text
Linear(input_dim, 256)
SiLU
Linear(256, 256)
SiLU
Linear(256, 256)
SiLU
Linear(256, C)
```

Weight layout:

```text
W: [out_dim, in_dim]
b: [out_dim]
```

Linear scalar rule:

```text
Y[r, o] = b[o] + sum_i W[o, i] * X[r, i]
```

SiLU:

```text
silu(x) = x * sigmoid(x)
sigmoid(x) = 1 / (1 + exp(-x))
d_silu(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
```

## Final MSE Mode

MSE mode predicts target values directly from embedded coordinates.

```text
features = fourier_embedding(coordinates, F)
prediction = mlp(features)
loss_target = targets
loss = mean((prediction - targets)^2)
```

Dimensions:

```text
F = 8:  input_dim = 34, output_dim = 3
F = 16: input_dim = 66, output_dim = 3
```

Loss:

```text
R = B * N
count = R * C
loss_mse = (1 / count) * sum((prediction - targets)^2)
d_prediction = 2 * (prediction - targets) / count
```

## Final CICFM Mode

CICFM mode predicts value-space velocity.

```text
z1 = targets
z0 ~ Uniform(0, 1)
t ~ Uniform(0, 1), shape [B, N, 1]
zt = (1 - t) * z0 + t * z1
target_velocity = z1 - z0
cicfm_input = concat(features, zt, t)
predicted_velocity = mlp(cicfm_input)
loss = mean((predicted_velocity - target_velocity)^2)
```

Dimensions:

```text
F = 8:  input_dim = 38, output_dim = 3
F = 16: input_dim = 70, output_dim = 3
```

Output gradient:

```text
R = B * N
count = R * C
d_predicted_velocity =
  2 * (predicted_velocity - target_velocity) / count
```

CICFM uses the same MSE reduction and MLP backward implementation. It changes
which tensors are passed into that machinery.

## Explicit Loss Mode

The training API must select the objective explicitly:

```text
loss_mode = mse
loss_mode = cicfm
```

The mode controls:

1. Model input assembly.
2. Output semantic name.
3. Loss target selection.
4. First layer input dimension.
5. Mode-specific logging fields.

The mode must not change:

```text
hidden width
hidden layer count
SiLU activation
identity output
optimizer update rule
MSE reduction formula
linear backward formula
gradient clearing rule
```

## Coding Plan Overview

The coding phase should land in small PRs. Each PR should compile and test
independently.

| PR | Owner Track | Goal | Main Files |
| --- | --- | --- | --- |
| 1 | Shared foundation | Add model/loss headers and configs | `include/model/*`, `src/model/*` |
| 2 | Model math | Implement MLP forward and SiLU | `include/model/mlp.h`, `src/model/mlp.cpp`, `tests/test_mlp_forward.cpp` |
| 3 | Loss math | Implement MSE loss and gradients | `include/training/loss.h`, `src/training/loss.cpp`, `tests/test_loss.cpp` |
| 4 | CICFM math | Implement CICFM batch assembly | `include/training/cicfm.h`, `src/training/cicfm.cpp`, `tests/test_cicfm.cpp` |
| 5 | Backward math | Implement manual MLP backward | `src/model/mlp.cpp`, `tests/test_mlp_backward.cpp` |
| 6 | Runtime | Add initialization, optimizer, and train loop | `include/training/*`, `src/training/*` |
| 7 | Debugging | Add finite checks and transition logs | `src/training/*`, `tests/test_training_debug.cpp` |
| 8 | Benchmarks | Add training/assembly timing fields | `benchmarks/*`, `benchmarks/results/*` |
| 9 | CUDA alignment | Keep embedding parity with model input | `src/kernels/*`, `tests/test_cpu_cuda_parity.cu` |

## Recommended File Layout

Add these files when the coding phase begins:

```text
include/model/mlp.h
src/model/mlp.cpp
include/training/loss.h
src/training/loss.cpp
include/training/cicfm.h
src/training/cicfm.cpp
include/training/train_config.h
include/training/train_debug.h
src/training/train_debug.cpp
tests/test_mlp_forward.cpp
tests/test_loss.cpp
tests/test_cicfm.cpp
tests/test_mlp_backward.cpp
tests/test_training_debug.cpp
```

Do not put model math inside benchmark files. Benchmarks should call the same
model and training helpers that tests call.

## Model Math Implementation Tasks

Your model-math track should implement these in order:

1. `ActivationKind` with `SiLU` as the only required baseline activation.
2. `MlpConfig` with `input_dim`, `hidden_width`, `output_dim`, and
   `hidden_layer_count`.
3. Parameter containers for four linear layers.
4. Forward cache containing `X`, `Z0`, `A0`, `Z1`, `A1`, `Z2`, `A2`, and `Y`.
5. `linear_forward`.
6. `silu_forward`.
7. `mlp_forward`.
8. `linear_backward`.
9. `silu_backward`.
10. `mlp_backward`.

Required cache rule:

```text
Backward must use tensors saved by the matching forward pass.
Do not recompute activations from parameters that may have already changed.
```

## Loss Implementation Tasks

Implement loss code as reusable helpers:

```text
mse_loss(output, target)
mse_output_gradient(output, target)
select_loss_tensors(loss_mode, batch, model_output)
```

For MSE:

```text
output = prediction
target = targets
```

For CICFM:

```text
output = predicted_velocity
target = target_velocity
```

Add a test that fails if CICFM compares `predicted_velocity` to `targets`.

## CICFM Assembly Tasks

Implement CICFM assembly separately from the MLP:

```text
make_cicfm_batch(features, targets, z0, t)
sample_z0(shape, seed)
sample_t(shape, seed)
```

Assembly formula:

```text
z1 = targets
zt = (1 - t) * z0 + t * z1
target_velocity = z1 - z0
cicfm_input = concat(features, zt, t)
```

Deterministic debug helper:

```text
make_deterministic_cicfm_batch(features, targets):
  z0 = zeros_like(targets)
  t = full([B, N, 1], 0.5)
```

## Shape Checks to Code

Add runtime assertions before every train step:

```text
coordinates.ndim == 3
features.ndim == 3
targets.ndim == 3
features.shape[0] == targets.shape[0]
features.shape[1] == targets.shape[1]
targets.shape[2] == C
```

MSE-specific:

```text
model_input.shape == features.shape
prediction.shape == targets.shape
```

CICFM-specific:

```text
z0.shape == targets.shape
z1.shape == targets.shape
t.shape == [B, N, 1]
zt.shape == targets.shape
target_velocity.shape == targets.shape
cicfm_input.shape[2] == feature_dim + C + 1
predicted_velocity.shape == target_velocity.shape
```

## Test Plan

Add tests in this order:

1. Fourier feature input dimension matches `D + 2 * D * F`.
2. MLP forward produces `[B, N, C]`.
3. SiLU forward matches hand-computed values.
4. MSE scalar loss matches hand-computed values.
5. MSE output gradient matches hand-computed values.
6. Linear backward matches hand-computed values.
7. SiLU backward matches hand-computed values.
8. Tiny MLP finite-difference gradient check passes.
9. Synthetic MSE overfit loss decreases.
10. CICFM deterministic assembly matches expected `zt`.
11. CICFM deterministic assembly matches expected `target_velocity`.
12. CICFM input dimension is `38` for `F = 8`.
13. CICFM input dimension is `70` for `F = 16`.
14. CICFM loss uses `target_velocity`, not `targets`.
15. CICFM deterministic forward/backward gradients are finite.
16. CICFM deterministic overfit loss decreases.
17. Random CICFM batch has finite loss and gradients.

## Benchmark Plan

Keep existing coordinate embedding benchmarks active. Add training benchmarks
only after the MSE overfit test passes.

Required benchmark fields:

```text
benchmark_name
loss_mode
device
batch_size
num_points
feature_dim
input_dim
output_dim
fourier_frequencies
forward_ms
loss_ms
backward_ms
step_ms
mean_loss
all_finite
```

CICFM-specific benchmark fields:

```text
cicfm_assembly_ms
mean_t
mean_abs_target_velocity
```

Do not compare CPU and CUDA training performance until CPU and CUDA embedding
inputs are parity-checked against the same model input contract.

## CUDA Alignment Plan

The model input contract is:

```text
features = concat(raw_coordinates, fourier_features)
cicfm_input = concat(features, zt, t)
```

The CUDA kernel currently covers the Fourier-feature portion. CUDA-facing
integration must prove:

```text
raw coordinates are present
Fourier feature order matches CPU
zt is appended after features
t is appended after zt
final input dimension matches loss_mode
```

Add parity checks before adding CUDA model kernels.

## Debug Gates

Use these gates during implementation:

```text
Gate 1: MSE forward shapes pass
Gate 2: MSE loss and output gradient pass hand checks
Gate 3: MLP backward passes finite differences
Gate 4: MSE tiny overfit loss decreases
Gate 5: deterministic CICFM assembly passes
Gate 6: CICFM loss target probe passes
Gate 7: CICFM deterministic backward gradients are finite
Gate 8: CICFM deterministic overfit loss decreases
Gate 9: random CICFM remains finite
Gate 10: F = 16 path passes the same shape and finite checks
```

Do not start a later gate until the earlier gate passes.

## Definition of Done for the Coding Phase

The model/loss coding phase is complete when:

1. MLP forward works for MSE and CICFM input dimensions.
2. SiLU forward and backward are tested.
3. Shared MSE loss and output gradient are tested.
4. Manual MLP backward passes finite-difference checks on a tiny model.
5. MSE overfit test decreases loss on a deterministic batch.
6. CICFM deterministic assembly passes value and shape checks.
7. CICFM loss uses `target_velocity`.
8. CICFM deterministic overfit decreases loss.
9. Random CICFM produces finite losses and gradients.
10. Logs include the Day 6 debug fields.
11. Benchmarks include enough fields to separate embedding, assembly, forward,
    loss, backward, and step timing.
12. CUDA embedding parity remains aligned with the model input contract.

## Day 7 Definition of Done

Day 7 is complete when:

1. The final MSE and CICFM design is consolidated in one document.
2. The exact tensor contract and dimensions are restated.
3. The coding plan names the modules and tests to add.
4. The implementation order is gated by correctness checks.
5. The benchmark and CUDA alignment plan are documented.
6. The model/loss contract points to this final design document.
