# Day 2: MLP Forward Semantics and MSE Target

This document freezes the Day 2 model-math contract for the first trainable
TinyINR baseline. It builds directly on `docs/model_loss_contract.md` and covers
only the MLP forward semantics and the baseline MSE target.

Day 2 does not define optimizer updates, parameter initialization details,
training-loop plumbing, sampling, logging implementation, or CUDA changes. Those
remain partner-track runtime tasks.

## Goal

Define a deterministic CPU MLP forward pass that maps embedded coordinates to
target values:

```text
features -> mlp(features) -> prediction
prediction vs targets -> MSE loss
```

For the first RGB INR baseline:

```text
features:   [B, N, input_dim]
targets:    [B, N, C]
prediction: [B, N, C]
```

The first training success criterion is overfitting a tiny fixed coordinate
batch with MSE before converting the model to CICFM.

## Input Feature Contract

The MLP does not consume raw coordinates directly. It consumes the output of the
Day 1 feature contract:

```text
features = concat(raw_coordinates, fourier_features)
```

Shape:

```text
features: [B, N, D + 2 * D * F]
```

For the first image baseline:

```text
D = 2
C = 3
F = 8 initially
```

Therefore:

```text
input_dim = D + 2 * D * F
input_dim = 2 + 2 * 2 * 8
input_dim = 34
```

The `F = 16` follow-up configuration is:

```text
input_dim = 2 + 2 * 2 * 16
input_dim = 66
```

## Batch Flattening Rule

For implementation simplicity, the MLP may treat `[B, N, input_dim]` as a 2D
matrix:

```text
rows = B * N
x_flat: [rows, input_dim]
```

The output is computed as:

```text
y_flat: [rows, C]
```

and then interpreted back as:

```text
prediction: [B, N, C]
```

This flattening is only a view of the logical training examples. It must not
change element order.

Flat row mapping:

```text
row = b * N + n
feature_index = (b * N + n) * input_dim + i
target_index  = (b * N + n) * C + c
```

## Architecture

The baseline model has four linear layers total:

```text
Linear(input_dim, 256)
SiLU
Linear(256, 256)
SiLU
Linear(256, 256)
SiLU
Linear(256, C)
```

This means there are three hidden layers and one output layer.

Use:

```text
hidden_width = 256
hidden_activation = SiLU
output_activation = identity
```

Do not apply sigmoid, tanh, clamp, or normalization to the final output in the
first baseline. If target image values are normalized to `[0, 1]`, the model is
still allowed to produce values outside `[0, 1]` during training.

## Parameter Shapes

Use row-major weight storage with shape:

```text
weight: [out_dim, in_dim]
bias:   [out_dim]
```

The scalar linear rule is:

```text
y[o] = bias[o] + sum_i weight[o, i] * x[i]
```

For `D = 2`, `F = 8`, and `C = 3`, the parameter shapes are:

| Layer | Input dim | Output dim | Weight shape | Bias shape |
| --- | ---: | ---: | --- | --- |
| `linear0` | 34 | 256 | `[256, 34]` | `[256]` |
| `linear1` | 256 | 256 | `[256, 256]` | `[256]` |
| `linear2` | 256 | 256 | `[256, 256]` | `[256]` |
| `linear3` | 256 | 3 | `[3, 256]` | `[3]` |

For `F = 16`, only the first layer changes:

```text
linear0.weight: [256, 66]
linear0.bias:   [256]
```

All other layer shapes stay the same.

## Forward Pass Semantics

For one flattened training row:

```text
h0_pre = linear0(features)
h0     = silu(h0_pre)

h1_pre = linear1(h0)
h1     = silu(h1_pre)

h2_pre = linear2(h1)
h2     = silu(h2_pre)

prediction = linear3(h2)
```

For a batch, apply this independently to every row in `B * N`.

Equivalent pseudocode:

```text
function mlp_forward(features):
    x = flatten_points(features)      # [B * N, input_dim]

    h0_pre = linear(x, W0, b0)        # [B * N, 256]
    h0 = silu(h0_pre)

    h1_pre = linear(h0, W1, b1)       # [B * N, 256]
    h1 = silu(h1_pre)

    h2_pre = linear(h1, W2, b2)       # [B * N, 256]
    h2 = silu(h2_pre)

    y = linear(h2, W3, b3)            # [B * N, C]
    return reshape(y, [B, N, C])
```

## SiLU Semantics

SiLU is applied elementwise to hidden pre-activations only.

Definition:

```text
silu(x) = x * sigmoid(x)
sigmoid(x) = 1 / (1 + exp(-x))
```

Derivative for backward verification:

```text
d_silu(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
```

Numerical note:

```text
Use float32 for consistency with the Tensor class.
```

## Forward Cache Requirements

The forward pass must save enough intermediate values for manual backward
implementation on Day 4. The minimum cache is:

```text
features
h0_pre
h0
h1_pre
h1
h2_pre
h2
prediction
```

The final output layer has no activation, so it does not need an output
pre-activation separate from `prediction`.

Recommended cache names:

```text
layer0_input = features_flat
layer0_pre
layer0_output
layer1_pre
layer1_output
layer2_pre
layer2_output
prediction_flat
```

## MSE Target Contract

In MSE mode, the target is the ground-truth value from `CoordinateBatch`:

```text
targets: [B, N, C]
```

For RGB reconstruction:

```text
C = 3
targets[b, n, 0] = red
targets[b, n, 1] = green
targets[b, n, 2] = blue
```

The model output must match the target shape exactly:

```text
prediction.shape == targets.shape
prediction: [B, N, C]
targets:    [B, N, C]
```

## MSE Loss Semantics

Use mean squared error over all batch, point, and channel elements:

```text
loss_mse = mean((prediction - targets)^2)
```

Expanded:

```text
count = B * N * C
loss_mse = (1 / count) * sum_b sum_n sum_c
           (prediction[b, n, c] - targets[b, n, c])^2
```

Gradient entering the output layer:

```text
dL/dprediction[b, n, c] =
    2 * (prediction[b, n, c] - targets[b, n, c]) / count
```

This gradient shape is:

```text
[B, N, C]
```

or flattened:

```text
[B * N, C]
```

## Required Shape Checks

The MSE path should reject invalid shapes early.

Required checks:

```text
features.ndim == 3
targets.ndim == 3
features.shape[0] == targets.shape[0]
features.shape[1] == targets.shape[1]
features.shape[2] == input_dim
targets.shape[2] == output_dim
```

For the first RGB baseline:

```text
input_dim == 34 when F = 8
output_dim == 3
```

## Overfit Target

The first training test should be intentionally tiny and deterministic.

Recommended synthetic target:

```text
y_r = 0.5 + 0.5 * x
y_g = 0.5 + 0.5 * y
y_b = 0.5 + 0.25 * sin(pi * x) + 0.25 * cos(pi * y)
```

where:

```text
x = coordinates[..., 0]
y = coordinates[..., 1]
```

Use a small grid first:

```text
B = 1
H = 8
W = 8
N = 64
D = 2
C = 3
F = 8
```

This target is better than starting with a real image because it is
deterministic, easy to inspect, and known to be representable by coordinate
features.

After the synthetic overfit works, repeat with one tiny real image or image
patch.

## Success Criteria for Day 2

Day 2 is complete when the following are unambiguous:

1. The MLP takes `features` with shape `[B, N, input_dim]`.
2. The MLP outputs `prediction` with shape `[B, N, C]`.
3. The architecture is exactly four linear layers total with width 256.
4. SiLU is applied after the first three linear layers only.
5. The final layer has identity output activation.
6. The weight layout is `[out_dim, in_dim]`.
7. The MSE target is `targets`, not velocity or noise.
8. MSE reduces over `B * N * C`.
9. The output gradient formula is frozen for Day 4 backward checks.
10. A deterministic tiny overfit target is defined for partner-track testing.

## Not in Day 2

Do not add these decisions to the Day 2 scope:

```text
optimizer type
learning rate schedule
parameter initialization scale
manual backward implementation
CICFM target assembly
CUDA feature concatenation changes
image loading format
adaptive sampling
```

Those are separate tasks. Day 2 is only the forward semantics and MSE target.
