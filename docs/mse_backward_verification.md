# Day 4: Verify MSE Training Logic and Backward Math

## Goal

Verify that the direct coordinate-to-value baseline has correct training math:

```text
features -> MLP -> prediction
prediction + targets -> MSE loss
MSE gradient -> output layer
manual backward -> all layer gradients
tiny fixed batch -> decreasing loss
```

The Day 4 output is a mathematical and testing contract. It defines what the
partner implementation must compute and how to check it.

## Forward Contract Recap

Flatten the logical `[B, N, input_dim]` feature tensor into:

```text
R = B * N
X: [R, input_dim]
```

The MLP is:

```text
Z0 = linear(X, W0, b0)
A0 = silu(Z0)

Z1 = linear(A0, W1, b1)
A1 = silu(Z1)

Z2 = linear(A1, W2, b2)
A2 = silu(Z2)

Y = linear(A2, W3, b3)
prediction = reshape(Y, [B, N, C])
```

Weight layout is fixed as:

```text
W: [out_dim, in_dim]
b: [out_dim]
```

Linear scalar rule:

```text
Y[r, o] = b[o] + sum_i W[o, i] * X[r, i]
```

## Required Forward Cache

Manual backward must save these tensors from the forward pass:

```text
X
Z0
A0
Z1
A1
Z2
A2
Y
targets_flat
loss
```

Do not recompute activations from changed parameters during backward. Backward
must use the values cached during the same forward pass.

## MSE Loss

Use mean squared error over every flattened example and output channel:

```text
R = B * N
count = R * C
loss = (1 / count) * sum_r sum_c (Y[r, c] - T[r, c])^2
```

where:

```text
Y: [R, C]
T: [R, C]
```

Gradient entering the output layer:

```text
dY[r, c] = 2 * (Y[r, c] - T[r, c]) / count
```

Shape:

```text
dY: [R, C]
```

## Linear Backward

For:

```text
Y = X * W^T + b
```

with:

```text
X:  [R, in_dim]
W:  [out_dim, in_dim]
b:  [out_dim]
dY: [R, out_dim]
```

the gradients are:

```text
dW[o, i] = sum_r dY[r, o] * X[r, i]
db[o]    = sum_r dY[r, o]
dX[r, i] = sum_o dY[r, o] * W[o, i]
```

Shapes:

```text
dW: [out_dim, in_dim]
db: [out_dim]
dX: [R, in_dim]
```

These formulas apply to all four linear layers.

## SiLU Backward

SiLU is applied only after the first three linear layers:

```text
silu(x) = x * sigmoid(x)
sigmoid(x) = 1 / (1 + exp(-x))
```

Derivative:

```text
d_silu(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
```

For upstream gradient `dA`:

```text
dZ = dA * d_silu(Z)
```

Shape:

```text
dA.shape == Z.shape
dZ.shape == Z.shape
```

Use the cached pre-activation `Z`, not the post-activation `A`, to compute the
derivative.

## Full MLP Backward Order

Backward starts from the MSE output gradient:

```text
dY = 2 * (Y - T) / (R * C)
```

Then apply:

```text
dW3, db3, dA2 = linear_backward(dY, A2, W3)

dZ2 = dA2 * d_silu(Z2)
dW2, db2, dA1 = linear_backward(dZ2, A1, W2)

dZ1 = dA1 * d_silu(Z1)
dW1, db1, dA0 = linear_backward(dZ1, A0, W1)

dZ0 = dA0 * d_silu(Z0)
dW0, db0, dX = linear_backward(dZ0, X, W0)
```

Required gradient outputs:

```text
dW0, db0
dW1, db1
dW2, db2
dW3, db3
```

`dX` is useful for debugging but is not a trainable parameter gradient.

## Gradient Accumulation Rule

Each backward call computes gradients for exactly one forward pass. Gradients
must be cleared before each new backward pass unless the runtime explicitly
implements gradient accumulation.

Day 4 baseline rule:

```text
zero_grad()
forward()
loss()
backward()
optimizer_step()
```

Do not accumulate gradients across steps in the first implementation.

## Numerical Gradient Check

Use central differences for gradient checks:

```text
numeric_grad(theta_i) =
    (loss(theta_i + eps) - loss(theta_i - eps)) / (2 * eps)
```

Recommended epsilon:

```text
eps = 1e-3
```

Recommended tolerances for float32:

```text
absolute_error < 1e-3
relative_error < 1e-2
```

Relative error:

```text
relative_error =
    abs(analytic - numeric) / max(1e-8, abs(analytic) + abs(numeric))
```

Gradient-check only a small subset of parameters. Full 256-wide layers are too
large for exhaustive finite differences.

Recommended gradient-check model:

```text
input_dim = 5
hidden_width = 4
output_dim = 3
R = 2
```

This tiny model uses the same formulas but keeps finite differences cheap.

## Unit Verification Gates

The partner implementation should add checks in this order:

1. MSE scalar value check.
2. MSE output-gradient check.
3. SiLU derivative check.
4. Linear backward shape and value check.
5. Full MLP backward finite-difference check.
6. Tiny fixed-batch training-loss decrease check.

Do not skip earlier gates if a later one fails. The failure will be harder to
diagnose.

## Gate 1: MSE Scalar Value

Use tiny hand-computable tensors:

```text
Y = [[1, 2, 3]]
T = [[0, 2, 4]]
```

Then:

```text
diff = [[1, 0, -1]]
squared = [[1, 0, 1]]
loss = (1 + 0 + 1) / 3 = 0.6666667
```

Expected:

```text
abs(loss - 0.6666667) < 1e-6
```

## Gate 2: MSE Output Gradient

For the same tensors:

```text
dY = 2 * (Y - T) / 3
dY = [[0.6666667, 0, -0.6666667]]
```

Expected:

```text
max_abs(dY - expected_dY) < 1e-6
```

## Gate 3: SiLU Derivative

Check analytic derivative against finite differences:

```text
silu(x) = x * sigmoid(x)
d_silu(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
```

Test points:

```text
x = -3, -1, 0, 1, 3
```

Expected:

```text
absolute_error < 1e-4
```

At `x = 0`:

```text
sigmoid(0) = 0.5
d_silu(0) = 0.5
```

## Gate 4: Linear Backward

Use a tiny layer:

```text
X:  [2, 3]
W:  [2, 3]
b:  [2]
dY: [2, 2]
```

Verify:

```text
dW[o, i] = sum_r dY[r, o] * X[r, i]
db[o] = sum_r dY[r, o]
dX[r, i] = sum_o dY[r, o] * W[o, i]
```

Expected:

```text
dW.shape == W.shape
db.shape == b.shape
dX.shape == X.shape
```

Then compare selected entries to hand-computed values.

## Gate 5: Full MLP Finite-Difference Check

Use a tiny version of the MLP:

```text
input_dim = 5
hidden_width = 4
output_dim = 3
R = 2
```

Use deterministic parameters and inputs. Recommended parameter values:

```text
small fixed values in [-0.1, 0.1]
```

Check only a subset:

```text
W0[0, 0]
W1[1, 2]
W2[3, 1]
W3[2, 0]
b0[0]
b1[1]
b2[2]
b3[0]
```

For each parameter:

```text
analytic = backward_gradient(parameter)
numeric = central_difference(parameter)
```

Pass criteria:

```text
absolute_error < 1e-3
relative_error < 1e-2
```

## Gate 6: Tiny Fixed-Batch Loss Decrease

Use the deterministic Day 2 synthetic target:

```text
y_r = 0.5 + 0.5 * x
y_g = 0.5 + 0.5 * y
y_b = 0.5 + 0.25 * sin(pi * x) + 0.25 * cos(pi * y)
```

Recommended tiny setup:

```text
B = 1
H = 8
W = 8
N = 64
D = 2
C = 3
F = 8
```

Expected behavior:

```text
loss_after_100_steps < loss_at_step_0
loss decreases on most steps, but not necessarily every step
```

Stronger overfit target:

```text
loss_after_training < 1e-3
```

This stronger target can wait until initialization and optimizer are stable.

## Required Training Logic Checks

Before trusting a training run, verify:

```text
features.shape == [B, N, input_dim]
targets.shape == [B, N, C]
prediction.shape == [B, N, C]
loss is finite
all gradients are finite
all parameter updates are finite
loss at step 0 is recorded before the first update
gradients are zeroed before each backward pass
```

If any value is `NaN` or `Inf`, stop the run immediately and print the failing
tensor name.

## Logging Requirements

For MSE training, log:

```text
step
loss_mode = mse
loss
batch_size
num_points
input_dim
output_dim
learning_rate
mean_abs_prediction
mean_abs_target
mean_abs_output_gradient
max_abs_gradient
```

Optional but useful:

```text
mean_abs_update
max_abs_update
```

## Common Failure Modes

| Symptom | Likely cause | First check |
| --- | --- | --- |
| Loss does not decrease at all | Sign error in gradients or optimizer update | Check one finite-difference parameter |
| Loss explodes immediately | Learning rate too high or initialization too large | Check update magnitude |
| Final-layer gradients pass but lower layers fail | SiLU derivative or cache bug | Check `d_silu(Z)` |
| Bias gradients are wrong | Summing over wrong axis | Check `db[o] = sum_r dY[r, o]` |
| Weight gradients transposed | Confused weight layout | Confirm `W[out_dim, in_dim]` |
| Shape works for `B = 1` only | Flattening bug | Check `row = b * N + n` |

## Not in Day 4

Do not include these in Day 4 scope:

```text
CICFM loss conversion
CICFM backward special cases
adaptive sampling
CUDA model kernels
ODE sampling
activation ablations
optimizer comparisons
```

Day 4 is only MSE training-logic verification and backward math.

## Day 4 Definition of Done

Day 4 is complete when:

1. MSE scalar loss formula is verified with a hand-computable example.
2. MSE output gradient is verified.
3. SiLU derivative is verified.
4. Linear backward formulas are frozen and shape-checked.
5. Full MLP backward order is frozen.
6. Finite-difference gradient-check protocol is documented.
7. Tiny fixed-batch loss-decrease criteria are documented.
8. Training logic checks and logging fields are documented.
9. Failure modes are listed with first checks.
10. CICFM remains blocked until MSE backward and tiny overfit checks pass.
