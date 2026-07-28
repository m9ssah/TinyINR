# Model Contract Guide

The baseline model includes: 
- 4 linear layers
- Width 256
- Depth 4
- SiLU activations
- no output activation 

**Architecture**: input -> 256 -> 256 -> 256 -> C

First objective: MSE 
- predict RGB value directly from embedded coordinates

Second objective: CICFM 
- predict velocity from coordinate, time, and interpolated value 

## Frozen Model Contract 
coordinates: [B, N, D]
targets:     [B, N, C]
features:    [B, N, D + 2 * D * F]
prediction:  [B, N, C]

**MSE Baseline Semantics**
gamma(x) = concat(
  x,
  sin(pi * 2^0 * x), cos(pi * 2^0 * x),
  sin(pi * 2^1 * x), cos(pi * 2^1 * x),
  ...
  sin(pi * 2^(F-1) * x), cos(pi * 2^(F-1) * x)
)

y_hat = mlp(gamma(x))
loss_mse = mean((y_hat - y)^2)


**CICFM Semantics**
z0 = random source value, shape [B, N, C]
z1 = target value, shape [B, N, C]
t  = uniform time in [0, 1], broadcast to [B, N, 1]

zt = (1 - t) * z0 + t * z1
target_velocity = z1 - z0

model_input = concat(gamma(x), zt, t)
pred_velocity = mlp(model_input)
loss_cicfm = mean((pred_velocity - target_velocity)^2)

# To Do: 
- [] Freeze model and loss contract
- [] Define MLP forward semantics and MSE target
- [] Define CICFM Scaffolding 
- [] Verify MSE training logic and backward math 
- [] Convert MSE to CICFM (use explicit loss mode)
- [] Debug model behavior and loss transition 
- [] Document the final model and loss design 
 

