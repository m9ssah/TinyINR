# Day 3: CICFM Scaffolding

CICFM scaffolding defines the tensors and math needed for the conversion from direct value prediction to velocity prediction.

## Goal

Define the CICFM batch assembly math:

```text
targets -> z1
noise/source values -> z0
time samples -> t
interpolated values -> zt
velocity target -> target_velocity
model input -> cicfm_input
model output -> predicted_velocity
loss target -> target_velocity
```

The model still uses the same hidden MLP architecture:

```text
Linear(input_dim, 256)
SiLU
Linear(256, 256)
SiLU
Linear(256, 256)
SiLU
Linear(256, C)
```

Only the input dimension and output semantics change.

## Tensor Names and Shapes

Use these exact tensor names in code, tests, logs, and docs:

| Name | Meaning | Shape |
| --- | --- | --- |
| `coordinates` | Normalized coordinate positions | `[B, N, D]` |
| `targets` | Ground-truth values from `CoordinateBatch` | `[B, N, C]` |
| `features` | Raw coordinates plus Fourier features | `[B, N, D + 2 * D * F]` |
| `z0` | Source/noise value | `[B, N, C]` |
| `z1` | Data/target value | `[B, N, C]` |
| `t` | Time conditioning scalar | `[B, N, 1]` |
| `zt` | Interpolated value at time `t` | `[B, N, C]` |
| `target_velocity` | Velocity regression target | `[B, N, C]` |
| `cicfm_input` | MLP input in CICFM mode | `[B, N, D + 2 * D * F + C + 1]` |
| `predicted_velocity` | MLP output in CICFM mode | `[B, N, C]` |

For the first RGB image baseline:

```text
B = batch size
N = sampled coordinate count
D = 2
C = 3
F = 8 first, then F = 16
```

## Source and Target Definitions

The target endpoint is the observed value:

```text
z1 = targets
```

For RGB reconstruction:

```text
z1[b, n, 0] = red target
z1[b, n, 1] = green target
z1[b, n, 2] = blue target
```

The source endpoint is random noise or a simple source distribution in value
space:

```text
z0 ~ Uniform(0, 1)
```

Use `Uniform(0, 1)` for the first implementation because image targets are
expected to live in `[0, 1]`. 

Required shape:

```text
z0.shape == targets.shape
z1.shape == targets.shape
```

Recommended deterministic debugging mode:

```text
z0 = zeros_like(targets)
```

This mode is useful for checking interpolation and velocity formulas before
turning random sampling on.

## Time Sampling

Use per-point time samples:

```text
t ~ Uniform(0, 1)
t.shape = [B, N, 1]
```

Per-point time gives each batch broad coverage of the interpolation path.

Do not use sinusoidal time embeddings in the first CICFM implementation. The
time value is concatenated as one raw scalar.

Recommended deterministic debugging mode:

```text
t = 0.5 for every point
```

Required time checks:

```text
t.ndim == 3
t.shape == [B, N, 1]
0 <= t[b, n, 0] <= 1
```

## Interpolation

Use the linear interpolation path:

```text
zt = (1 - t) * z0 + t * z1
```

Since `t` has shape `[B, N, 1]`, it broadcasts across channels:

```text
zt[b, n, c] =
    (1 - t[b, n, 0]) * z0[b, n, c]
  + t[b, n, 0] * z1[b, n, c]
```

Endpoint sanity checks:

```text
if t = 0, then zt = z0
if t = 1, then zt = z1
if t = 0.5, then zt = 0.5 * z0 + 0.5 * z1
```

## Velocity Target

The target velocity for this linear path is:

```text
target_velocity = z1 - z0
```

Expanded:

```text
target_velocity[b, n, c] = z1[b, n, c] - z0[b, n, c]
```

This target is independent of `t` for the first linear CICFM path. That is
intentional.

Required shape:

```text
target_velocity.shape == [B, N, C]
```

## CICFM Model Input Assembly

The model input is:

```text
cicfm_input = concat(features, zt, t)
```

Concatenate along the last dimension.

For each point:

```text
cicfm_input[b, n] = [
  features[b, n, :],
  zt[b, n, :],
  t[b, n, 0]
]
```

Shape:

```text
cicfm_input.shape = [B, N, feature_dim + C + 1]
feature_dim = D + 2 * D * F
```

For `D = 2`, `C = 3`, and `F = 8`:

```text
feature_dim = 2 + 2 * 2 * 8 = 34
cicfm_input_dim = 34 + 3 + 1 = 38
output_dim = 3
```

For `F = 16`:

```text
feature_dim = 2 + 2 * 2 * 16 = 66
cicfm_input_dim = 66 + 3 + 1 = 70
output_dim = 3
```

## CICFM Forward Semantics

In MSE mode:

```text
prediction = mlp(features)
loss = mean((prediction - targets)^2)
```

In CICFM mode:

```text
predicted_velocity = mlp(cicfm_input)
loss = mean((predicted_velocity - target_velocity)^2)
```

The model output shape is still `[B, N, C]`, but its meaning changes:

```text
MSE output meaning: predicted value
CICFM output meaning: predicted velocity
```

Do not compare `predicted_velocity` directly to `targets`.

## Loss Formula

Use the same MSE reduction structure as Day 2, but with velocity tensors:

```text
count = B * N * C
loss_cicfm = (1 / count) * sum_b sum_n sum_c
             (predicted_velocity[b, n, c]
              - target_velocity[b, n, c])^2
```

Gradient entering the output layer:

```text
dL/dpredicted_velocity[b, n, c] =
    2 * (predicted_velocity[b, n, c]
         - target_velocity[b, n, c]) / count
```

## Required Shape Checks

Reject invalid CICFM batches early.

Required checks:

```text
features.ndim == 3
targets.ndim == 3
z0.ndim == 3
z1.ndim == 3
t.ndim == 3

features.shape[0] == targets.shape[0]
features.shape[1] == targets.shape[1]
z0.shape == targets.shape
z1.shape == targets.shape
t.shape[0] == targets.shape[0]
t.shape[1] == targets.shape[1]
t.shape[2] == 1

features.shape[2] == D + 2 * D * F
cicfm_input.shape[2] == D + 2 * D * F + C + 1
target_velocity.shape == targets.shape
predicted_velocity.shape == targets.shape
```

## Runtime Assembly Pseudocode

```text
function build_cicfm_batch(coordinates, targets, F):
    features = fourierEmbedding(coordinates, F)
    z1 = targets
    z0 = sample_uniform_like(targets, low=0, high=1)
    t = sample_uniform([B, N, 1], low=0, high=1)

    zt = (1 - t) * z0 + t * z1
    target_velocity = z1 - z0
    cicfm_input = concat(features, zt, t, axis=-1)

    return {
        coordinates,
        targets,
        features,
        z0,
        z1,
        t,
        zt,
        target_velocity,
        cicfm_input
    }
```

## Deterministic Smoke Test

Before random sampling is enabled, use:

```text
z0 = zeros_like(targets)
t = 0.5
z1 = targets
```

Then:

```text
zt = 0.5 * targets
target_velocity = targets
cicfm_input = concat(features, 0.5 * targets, 0.5)
```

Expected checks:

```text
max_abs(zt - 0.5 * targets) < 1e-6
max_abs(target_velocity - targets) < 1e-6
cicfm_input.shape[2] == feature_dim + C + 1
```

This smoke test isolates tensor assembly from random number generation.

## Logging Requirements

For CICFM training logs, include:

```text
step
loss_mode = cicfm
loss
batch_size
num_points
input_dim
output_dim
learning_rate
mean_t
mean_abs_zt
mean_abs_target_velocity
mean_abs_predicted_velocity
```

Optional but useful debug fields:

```text
min_t
max_t
mean_abs_z0
mean_abs_z1
```

## Not in Day 3

Do not include these in the Day 3 scope:

```text
optimizer implementation
parameter initialization
manual backward implementation
CICFM training behavior analysis
ODE sampling after training
separate context tensor
sinusoidal time embedding
CUDA kernel changes
adaptive coordinate sampling
```

Day 3 is only the CICFM tensor scaffolding and target construction contract.

## Day 3 Definition of Done

Day 3 is complete when:

1. `z0`, `z1`, `t`, `zt`, and `target_velocity` are defined with exact shapes.
2. `z0` source distribution is frozen as `Uniform(0, 1)` for the first version.
3. Time conditioning is frozen as per-point scalar `t` with shape `[B, N, 1]`.
4. Interpolation is frozen as `zt = (1 - t) * z0 + t * z1`.
5. The velocity target is frozen as `target_velocity = z1 - z0`.
6. CICFM model input is frozen as `concat(features, zt, t)`.
7. CICFM input dimensions are documented for `F = 8` and `F = 16`.
8. The CICFM loss formula and output gradient are documented.
9. Required shape checks are listed.
10. A deterministic assembly smoke test is defined.
