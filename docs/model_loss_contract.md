# TinyINR Model and Loss Contract

This document specifies the model and loss design for the first trainable
TinyINR baseline. The goal is to make the learning problem unambiguous before
the training loop, optimizer, sampler, and CUDA integration are implemented.

Day 2 extends this contract in `docs/mlp_forward_mse_semantics.md`, which
freezes the MLP forward pass and MSE target semantics.

Day 3 extends this contract in `docs/cicfm_scaffolding.md`, which freezes time
sampling, interpolation, velocity targets, and CICFM model input assembly.

Day 4 extends this contract in `docs/mse_backward_verification.md`, which
freezes MSE training-logic checks and manual backward math for the baseline MLP.

Day 5 extends this contract in `docs/mse_to_cicfm_conversion.md`, which freezes
the controlled conversion from direct MSE value prediction to CICFM velocity
prediction.

Day 6 extends this contract in `docs/loss_transition_debugging.md`, which
freezes the deterministic probes, logging, finite-value checks, and triage rules
used to debug the MSE-to-CICFM transition.

## Scope

The first training milestone has two controlled phases:

1. Train a direct coordinate-to-value INR with MSE.
2. Convert the same baseline into a CICFM velocity predictor.

The MSE path must work first. Do not debug CICFM until the MSE baseline can
overfit a tiny fixed batch.

## Naming

| Name | Meaning | Shape |
| --- | --- | --- |
| `coordinates` | Normalized input coordinates | `[B, N, D]` |
| `targets` | Ground-truth values at coordinates | `[B, N, C]` |
| `features` | Raw coordinates plus Fourier features | `[B, N, D + 2 * D * F]` |
| `prediction` | MSE-mode predicted values | `[B, N, C]` |
| `z0` | CICFM source/noise value | `[B, N, C]` |
| `z1` | CICFM target/data value, same as `targets` | `[B, N, C]` |
| `t` | CICFM time value | `[B, N, 1]` |
| `zt` | Interpolated CICFM value | `[B, N, C]` |
| `target_velocity` | CICFM regression target | `[B, N, C]` |
| `predicted_velocity` | CICFM model output | `[B, N, C]` |

For the first image baseline:

```text
B = batch size
N = sampled coordinate count
D = 2
C = 3
F = 8 initially, then F = 16
```

## Fourier Feature Contract

The model input must concatenate raw coordinates and Fourier features:

```text
features = concat(raw_coordinates, fourier_features)
```

For one scalar coordinate value `x`, the Fourier feature order is:

```text
sin(pi * 2^0 * x), cos(pi * 2^0 * x),
sin(pi * 2^1 * x), cos(pi * 2^1 * x),
...
sin(pi * 2^(F - 1) * x), cos(pi * 2^(F - 1) * x)
```

For one coordinate vector with dimension `D`, the full feature order is:

```text
x_0, x_1, ..., x_(D - 1),
freq_0 features for all D coordinate dimensions,
freq_1 features for all D coordinate dimensions,
...
freq_(F - 1) features for all D coordinate dimensions
```

The CPU tensor wrapper currently owns this full model feature contract. The
CUDA kernel currently outputs only the Fourier-feature portion with shape
`[B, N, D, F, 2]` flattened as `B * N * D * F * 2`. CUDA integration must either
concatenate raw coordinates outside the kernel or update the kernel contract in
a separate, parity-tested change.

## Baseline Architecture

Use one MLP class/configuration for both MSE and CICFM. The input dimension
changes by loss mode, but the hidden architecture stays fixed.

```text
Linear(input_dim, 256)
SiLU
Linear(256, 256)
SiLU
Linear(256, 256)
SiLU
Linear(256, output_dim)
```

Activation choice:

```text
hidden activation = SiLU
output activation = identity
```

SiLU is the baseline activation because TinyINR models continuous
coordinate-to-value functions. ReLU can be added later as an ablation.

SiLU definition:

```text
silu(x) = x * sigmoid(x)
sigmoid(x) = 1 / (1 + exp(-x))
```

SiLU derivative for backward checks:

```text
d/dx silu(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
```

## MSE Mode

MSE mode is the first trainable baseline. The model predicts target values
directly from embedded coordinates.

```text
features = fourierEmbedding(coordinates, F)
prediction = mlp(features)
loss = mean((prediction - targets)^2)
```

MSE model dimensions:

```text
input_dim = D + 2 * D * F
output_dim = C
```

For the first RGB image baseline with `D = 2`, `C = 3`, and `F = 8`:

```text
input_dim = 2 + 2 * 2 * 8 = 34
output_dim = 3
```

For `F = 16`:

```text
input_dim = 2 + 2 * 2 * 16 = 66
output_dim = 3
```

Loss reduction:

```text
loss_mse = (1 / (B * N * C)) * sum((prediction - targets)^2)
```

Gradient entering the final layer:

```text
dL/dprediction = 2 * (prediction - targets) / (B * N * C)
```

## CICFM Mode

CICFM mode converts the model from value prediction to velocity prediction.
It should be implemented only after the MSE path can overfit a tiny batch.

For each coordinate-value pair:

```text
z0 = random source/noise value
z1 = target value
t = uniform random time in [0, 1]
zt = (1 - t) * z0 + t * z1
target_velocity = z1 - z0
```

The model input is:

```text
cicfm_input = concat(features, zt, t)
```

The model output is:

```text
predicted_velocity = mlp(cicfm_input)
```

The CICFM loss is:

```text
loss_cicfm = mean((predicted_velocity - target_velocity)^2)
```

CICFM model dimensions:

```text
input_dim = D + 2 * D * F + C + 1
output_dim = C
```

For `D = 2`, `C = 3`, and `F = 8`:

```text
input_dim = 34 + 3 + 1 = 38
output_dim = 3
```

For `F = 16`:

```text
input_dim = 66 + 3 + 1 = 70
output_dim = 3
```

## Time Conditioning Decision

Use per-point time samples first:

```text
t shape = [B, N, 1]
```

`t` is concatenated as a scalar feature. Do not implement sinusoidal time
embedding in the first baseline.

## Context Decision

Do not add a separate context tensor in the first CICFM implementation.

The minimal CICFM input is:

```text
concat(features, zt, t)
```

If a future experiment needs explicit context, it should be added as:

```text
concat(features, zt, t, context)
```

but that is outside this first contract.

## Loss Mode Switch

Training should support an explicit loss mode:

```text
loss_mode = mse
loss_mode = cicfm
```

The mode controls only:

1. How the model input is assembled.
2. What the model output means.
3. Which target tensor is used in the MSE reduction.

It should not require a different optimizer, parameter update mechanism, or
logging system.

## Partner Runtime Requirements

The training loop should expose these fields for both modes:

```text
step
loss_mode
loss
batch_size
num_points
input_dim
output_dim
learning_rate
```

For CICFM, additionally log:

```text
mean_t
mean_abs_zt
mean_abs_target_velocity
mean_abs_predicted_velocity
```

For MSE, additionally log:

```text
mean_abs_prediction
mean_abs_target
```

# To Do:
- [x] Freeze model and loss contract
- [x] Define MLP forward semantics and MSE target
- [x] Define CICFM Scaffolding
- [x] Verify MSE training logic and backward math
- [x] Convert MSE to CICFM (use explicit loss mode)
- [x] Debug model behavior and loss transition
- [ ] Document the final model and loss design
 
