# Day 5: Convert MSE to CICFM

This document freezes the controlled conversion from the MSE baseline to the
CICFM velocity-prediction objective. It builds on:

- `docs/model_loss_contract.md`
- `docs/mlp_forward_mse_semantics.md`
- `docs/cicfm_scaffolding.md`
- `docs/mse_backward_verification.md`

Day 5 does not redesign the model. It changes what enters the model, what the
model output means, and what target the MSE reduction compares against.

## Goal

Convert the baseline from:

```text
features -> mlp -> predicted value
predicted value vs targets -> MSE
```

to:

```text
concat(features, zt, t) -> mlp -> predicted velocity
predicted velocity vs target_velocity -> MSE
```

The hidden MLP architecture, optimizer, parameter-update rule, batch flattening
rule, and backward formulas remain the same.

## Required Precondition

Do not enable CICFM training until the MSE path passes Day 4 checks:

```text
MSE scalar loss check passes
MSE output-gradient check passes
SiLU derivative check passes
linear backward check passes
full MLP finite-difference check passes
tiny fixed-batch MSE loss decreases
```

If the MSE path cannot overfit a deterministic fixed batch, CICFM training
should remain disabled.

## Explicit Loss Mode

Training must use an explicit mode:

```text
loss_mode = mse
loss_mode = cicfm
```

Do not infer the mode from tensor shapes. Shape inference makes debugging more
fragile once both objectives exist.

Recommended enum:

```text
enum LossMode {
  MSE,
  CICFM
}
```

The mode controls exactly three things:

1. How model input is assembled.
2. What the model output means.
3. Which target tensor is used by the MSE reduction.

The mode must not change:

```text
hidden layer count
hidden width
activation
output activation
optimizer type
parameter update formula
linear backward formula
SiLU backward formula
batch flattening rule
```

## MSE Mode Contract

MSE mode remains:

```text
model_input = features
model_output = prediction
loss_target = targets
loss = mean((prediction - targets)^2)
```

Shapes:

```text
features:    [B, N, D + 2 * D * F]
prediction:  [B, N, C]
targets:     [B, N, C]
```

For `D = 2`, `C = 3`, and `F = 8`:

```text
model_input_dim = 34
model_output_dim = 3
```

For `F = 16`:

```text
model_input_dim = 66
model_output_dim = 3
```

## CICFM Mode Contract

CICFM mode is:

```text
z1 = targets
z0 ~ Uniform(0, 1)
t  ~ Uniform(0, 1), shape [B, N, 1]
zt = (1 - t) * z0 + t * z1
target_velocity = z1 - z0

model_input = cicfm_input = concat(features, zt, t)
model_output = predicted_velocity
loss_target = target_velocity
loss = mean((predicted_velocity - target_velocity)^2)
```

Shapes:

```text
features:           [B, N, D + 2 * D * F]
z0:                 [B, N, C]
z1:                 [B, N, C]
t:                  [B, N, 1]
zt:                 [B, N, C]
target_velocity:    [B, N, C]
cicfm_input:        [B, N, D + 2 * D * F + C + 1]
predicted_velocity: [B, N, C]
```

For `D = 2`, `C = 3`, and `F = 8`:

```text
feature_dim = 34
cicfm_input_dim = 34 + 3 + 1 = 38
model_output_dim = 3
```

For `F = 16`:

```text
feature_dim = 66
cicfm_input_dim = 66 + 3 + 1 = 70
model_output_dim = 3
```

## Conversion Table

| Component | MSE mode | CICFM mode |
| --- | --- | --- |
| Model input | `features` | `concat(features, zt, t)` |
| Input dimension | `D + 2DF` | `D + 2DF + C + 1` |
| Model output name | `prediction` | `predicted_velocity` |
| Output meaning | Value/RGB | Velocity in value space |
| Output dimension | `C` | `C` |
| Loss target | `targets` | `target_velocity = z1 - z0` |
| Loss formula | `mean((prediction - targets)^2)` | `mean((predicted_velocity - target_velocity)^2)` |
| Backward formula | MSE output gradient | Same MSE output gradient, different tensors |

## Shared MSE Reduction

Both modes use the same MSE reduction helper:

```text
mse_loss(output, target) =
  (1 / count) * sum((output - target)^2)
```

The only difference is which tensors are passed to it.

MSE mode:

```text
output = prediction
target = targets
```

CICFM mode:

```text
output = predicted_velocity
target = target_velocity
```

Output gradient for both modes:

```text
d_output = 2 * (output - target) / count
```

For CICFM specifically:

```text
d_predicted_velocity =
  2 * (predicted_velocity - target_velocity) / (B * N * C)
```

## Backward Math Reuse

The full MLP backward order does not change:

```text
dY = mse_output_gradient(output, target)

dW3, db3, dA2 = linear_backward(dY, A2, W3)
dZ2 = dA2 * d_silu(Z2)
dW2, db2, dA1 = linear_backward(dZ2, A1, W2)
dZ1 = dA1 * d_silu(Z1)
dW1, db1, dA0 = linear_backward(dZ1, A0, W1)
dZ0 = dA0 * d_silu(Z0)
dW0, db0, dX = linear_backward(dZ0, X, W0)
```

What changes is:

```text
X = flatten(features)             in MSE mode
X = flatten(cicfm_input)          in CICFM mode
target = flatten(targets)         in MSE mode
target = flatten(target_velocity) in CICFM mode
```

## Model Construction Rule

Because the input dimension changes, the first layer shape changes between
modes.

MSE with `F = 8`:

```text
linear0.weight: [256, 34]
linear0.bias:   [256]
```

CICFM with `F = 8`:

```text
linear0.weight: [256, 38]
linear0.bias:   [256]
```

MSE with `F = 16`:

```text
linear0.weight: [256, 66]
linear0.bias:   [256]
```

CICFM with `F = 16`:

```text
linear0.weight: [256, 70]
linear0.bias:   [256]
```

The remaining layers are unchanged:

```text
linear1.weight: [256, 256]
linear2.weight: [256, 256]
linear3.weight: [C, 256]
```

Do not try to reuse an MSE model instance in CICFM mode unless the runtime
supports rebuilding or resizing the first layer. The clean first implementation
should construct a new model with the CICFM input dimension.

## Shape Checks

Required mode-independent checks:

```text
coordinates.ndim == 3
features.ndim == 3
targets.ndim == 3
features.shape[0] == targets.shape[0]
features.shape[1] == targets.shape[1]
targets.shape[2] == C
```

MSE-specific checks:

```text
model_input.shape == features.shape
model_input.shape[2] == D + 2 * D * F
model_output.shape == targets.shape
loss_target.shape == targets.shape
```

CICFM-specific checks:

```text
z0.shape == targets.shape
z1.shape == targets.shape
zt.shape == targets.shape
target_velocity.shape == targets.shape
t.shape == [B, N, 1]
cicfm_input.shape[0] == B
cicfm_input.shape[1] == N
cicfm_input.shape[2] == D + 2 * D * F + C + 1
predicted_velocity.shape == target_velocity.shape
```

Required value checks:

```text
0 <= t[b, n, 0] <= 1
loss is finite
all gradients are finite
```

## Deterministic Conversion Smoke Test

Before enabling random CICFM batches, use deterministic values:

```text
z0 = zeros_like(targets)
t = 0.5
z1 = targets
```

Expected:

```text
zt = 0.5 * targets
target_velocity = targets
cicfm_input = concat(features, 0.5 * targets, 0.5)
```

Checks:

```text
max_abs(zt - 0.5 * targets) < 1e-6
max_abs(target_velocity - targets) < 1e-6
cicfm_input.shape[2] == feature_dim + C + 1
```

If this deterministic smoke test fails, do not run random CICFM training.

## Controlled Training Transition

Run the transition in this order:

1. Train MSE mode on the deterministic synthetic overfit target.
2. Confirm MSE loss decreases.
3. Run deterministic CICFM batch assembly with `z0 = 0` and `t = 0.5`.
4. Confirm CICFM shape and value checks pass.
5. Run one CICFM forward pass without updating parameters.
6. Confirm CICFM loss is finite.
7. Run one CICFM backward pass.
8. Confirm all CICFM gradients are finite.
9. Only then run CICFM training updates.

## Logging Changes

Shared fields:

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

MSE fields:

```text
mean_abs_prediction
mean_abs_target
mean_abs_output_gradient
```

CICFM fields:

```text
mean_t
min_t
max_t
mean_abs_z0
mean_abs_z1
mean_abs_zt
mean_abs_target_velocity
mean_abs_predicted_velocity
mean_abs_output_gradient
```

The `mean_abs_output_gradient` field should refer to:

```text
d_prediction           in MSE mode
d_predicted_velocity   in CICFM mode
```

## Unit Test Plan

Add small tests in this order:

1. `LossMode` dispatch selects MSE tensors correctly.
2. `LossMode` dispatch selects CICFM tensors correctly.
3. CICFM deterministic assembly produces expected `zt`.
4. CICFM deterministic assembly produces expected `target_velocity`.
5. CICFM input dimension is correct for `F = 8`.
6. CICFM input dimension is correct for `F = 16`.
7. CICFM loss uses `target_velocity`, not `targets`.
8. CICFM output gradient matches `2 * (predicted_velocity - target_velocity) / count`.

## Common Conversion Bugs

| Symptom | Likely cause | First check |
| --- | --- | --- |
| CICFM input shape is too small | Forgot to append `zt` or `t` | Check last dimension |
| CICFM loss decreases toward targets | Compared velocity output to `targets` | Check loss target |
| First layer shape mismatch | Reused MSE model with CICFM input | Check `linear0.weight` |
| `zt` equals `targets` for all t | Interpolation ignores `z0` | Check `zt` formula |
| `target_velocity` changes with t | Used derivative of wrong path | Check `z1 - z0` |
| Training starts with NaN | Invalid time/source values or large updates | Check finite-value logs |

## Not in Day 5

Do not include these in Day 5 scope:

```text
ODE sampling after training
adaptive coordinate sampling
sinusoidal time embedding
separate context tensor
CUDA model kernels
optimizer comparisons
activation ablations
```

Day 5 is only the controlled objective conversion from MSE value prediction to
CICFM velocity prediction.

## Day 5 Definition of Done

Day 5 is complete when:

1. `loss_mode` explicitly selects MSE or CICFM.
2. MSE mode still uses `features -> prediction -> targets`.
3. CICFM mode uses `concat(features, zt, t) -> predicted_velocity`.
4. CICFM loss target is `target_velocity`, not `targets`.
5. CICFM target velocity is `z1 - z0`.
6. CICFM input dimensions are documented for `F = 8` and `F = 16`.
7. First-layer shape changes are documented.
8. Shared MSE reduction and backward reuse are documented.
9. Deterministic CICFM smoke test is documented.
10. CICFM training remains gated behind MSE overfit and finite-gradient checks.
