# Tensor Implementation Notes

This document explains the current TinyINR `Tensor` implementation in:

- `include/tensor.h`
- `src/tensor.cpp`

This Tensor is a small CPU-first float32 tensor that gives the rest of TinyINR a reliable foundation for coordinates, Fourier embeddings, sampling, and later CUDA kernels.

## What This Tensor Owns

The `Tensor` class owns three private fields:

```cpp
std::vector<float> data_;
std::vector<int64_t> shape_;
std::vector<int64_t> strides_;
```

### `data_`

`data_` is the actual memory buffer.

It is a flat `std::vector<float>`, which means all tensor values are stored in one contiguous 1D array. Using `std::vector` gives safe RAII memory management:

- memory is allocated automatically
- memory is freed automatically
- no manual `new`
- no manual `delete`
- copying a Tensor copies the vector safely

The first version only supports `float`, which maps to CPU float32 storage.

### `shape_`

`shape_` records the logical dimensions of the tensor.

Examples:

```cpp
Tensor x({2, 3});       // shape [2, 3]
Tensor coords({1, 1024, 2}); // shape [batch, points, coordinate_dim]
```

For TinyINR, the important shared conventions are:

```text
coordinates: [batch_size, num_points, coordinate_dim]
values:      [batch_size, num_points, value_dim]
embedding:   [batch_size, num_points, embedding_dim]
```

For one RGB image of size `32 x 32`:

```text
coordinates: [1, 1024, 2]
values:      [1, 1024, 3]
```

### `strides_`

`strides_` explains how to convert multidimensional indices into a flat `data_` index.

TinyINR uses contiguous row-major layout. In row-major layout, the last dimension is adjacent in memory.

For shape `[2, 3]`, the tensor looks like:

```text
[[a, b, c],
 [d, e, f]]
```

But `data_` stores it as:

```text
[a, b, c, d, e, f]
```

The strides are:

```text
shape:   [2, 3]
strides: [3, 1]
```

The offset formula is:

```cpp
offset = i * strides[0] + j * strides[1];
```

So:

```text
at({0, 0}) -> data_[0]
at({0, 1}) -> data_[1]
at({1, 0}) -> data_[3]
at({1, 2}) -> data_[5]
```

For shape `[B, N, D]`, strides become:

```text
[N * D, D, 1]
```

That means:

```cpp
offset = batch * (N * D) + point * D + channel;
```

This is exactly what TinyINR needs for coordinate tensors such as `[B, H * W, 2]`.

## Constructor

The constructor is:

```cpp
Tensor::Tensor(const std::vector<int64_t>& shape)
    : data_(compute_numel(shape), 0.0f),
      shape_(shape),
      strides_(compute_strides(shape)) {}
```

It does three things:

1. Computes the number of elements from the shape.
2. Allocates `data_` with that many floats, initialized to `0.0f`.
3. Stores `shape_` and computes `strides_`.

Example:

```cpp
Tensor x({2, 3});
```

This creates:

```text
shape_:   [2, 3]
strides_: [3, 1]
data_:    [0, 0, 0, 0, 0, 0]
```

## Shape Validation

`compute_numel` validates the shape before allocation.

Rules:

- shape cannot be empty
- every dimension must be positive
- the total element count must not overflow `int64_t`

So these are invalid:

```cpp
Tensor a({});
Tensor b({2, 0});
Tensor c({2, -3});
```

This matters because invalid shapes should fail early, before memory layout bugs spread into CoordinateBatch or FourierEmbedding.

## Accessing Raw Data

The class exposes:

```cpp
float* data();
const float* data() const;
```

These return pointers to the underlying contiguous buffer.

Example:

```cpp
Tensor x({2, 3});
x.data()[0] = 1.0f;
x.data()[5] = 6.0f;
```

Use `data()` when you need fast direct access or integration with low-level code. Use `at(...)` when you want safer multidimensional indexing.

## Safe Indexing

The public indexing methods are:

```cpp
float& at(const std::vector<int64_t>& indices);
const float& at(const std::vector<int64_t>& indices) const;
```

They call the private helper:

```cpp
int64_t offset(const std::vector<int64_t>& indices) const;
```

`offset` checks:

- the number of indices matches the number of dimensions
- each index is inside bounds

Then it computes:

```cpp
flat_index += indices[i] * strides_[i];
```

Example:

```cpp
Tensor x({2, 3});
x.at({1, 2}) = 5.0f;
```

For shape `[2, 3]` and strides `[3, 1]`:

```text
offset = 1 * 3 + 2 * 1 = 5
```

So this writes:

```cpp
data_[5] = 5.0f;
```

Invalid indexing throws an exception instead of silently returning bad memory.

## `zeros`

`zeros` is simple:

```cpp
Tensor Tensor::zeros(const std::vector<int64_t>& shape) {
    return Tensor(shape);
}
```

The constructor already zero-initializes the buffer, so `zeros` is just a named helper.

Example:

```cpp
Tensor x = Tensor::zeros({2, 3});
```

## `randn`

`randn` creates a tensor filled with normally distributed random values:

```cpp
Tensor x = Tensor::randn({2, 3}, 0.0f, 1.0f);
```

It uses:

- `std::random_device` for a seed
- `std::mt19937` as the generator
- `std::normal_distribution<float>` for samples

`stddev` must be non-negative.

This is useful for initializing weights or generating simple test tensors. It is intentionally not deterministic yet. If deterministic tests are needed later, add an overload that accepts a seed.

## `reshape`

`reshape` returns a copy of the tensor with a new shape:

```cpp
Tensor y = x.reshape({3, 2});
```

The important rule:

```text
old numel must equal new numel
```

So this is valid:

```cpp
Tensor x({2, 3}); // 6 elements
Tensor y = x.reshape({3, 2}); // also 6 elements
```

This is invalid:

```cpp
Tensor z = x.reshape({4, 2}); // 8 elements
```

The current implementation copies data:

```cpp
Tensor result(new_shape);
result.data_ = data_;
return result;
```

This is simpler than a view and safer for the first CPU implementation. Later, a true view would need shared storage, offset metadata, and careful lifetime rules.

## `row`

`row` returns a copy of one slice along the first dimension.

Example with a 2D tensor:

```cpp
Tensor x({2, 3});
Tensor r = x.row(1);
```

If `x` has shape `[2, 3]`, then `r` has shape `[3]`.

For a TinyINR coordinate tensor:

```cpp
Tensor coords({4, 1024, 2});
Tensor first_batch = coords.row(0);
```

`first_batch` has shape:

```text
[1024, 2]
```

This works because the tensor is contiguous row-major. One row along the first dimension occupies a contiguous block in memory.

The implementation computes:

```cpp
row_size = result.numel();
source_start = row_index * row_size;
```

Then it copies from:

```text
data_[source_start] to data_[source_start + row_size]
```

## Elementwise Operations

The current elementwise operations are:

```cpp
Tensor add(const Tensor& other) const;
Tensor multiply(const Tensor& other) const;
```

Both require exact shape equality.

Example:

```cpp
Tensor a({2, 3});
Tensor b({2, 3});

Tensor c = a.add(b);
Tensor d = a.multiply(b);
```

This is valid because both tensors have shape `[2, 3]`.

This is invalid:

```cpp
Tensor a({2, 3});
Tensor b({3});

Tensor c = a.add(b);
```

There is no broadcasting yet. That is deliberate. Broadcasting adds useful behavior, but it also adds complexity and more shape rules. The first version stays explicit.

## Matrix Multiplication

`matmul` currently supports only 2D matrix multiplication.

The rule is:

```text
[M, K] @ [K, N] = [M, N]
```

Example:

```cpp
Tensor a({2, 3});
Tensor b({3, 2});
Tensor c = a.matmul(b);
```

`c` has shape:

```text
[2, 2]
```

The formula is:

```text
c[i, j] = sum over k of a[i, k] * b[k, j]
```

The implementation uses three loops:

```cpp
for i in rows:
    for j in cols:
        sum = 0
        for k in inner:
            sum += left[i, k] * right[k, j]
        result[i, j] = sum
```

This is not optimized. That is fine for now. It is a correctness baseline.

Later, this baseline can be compared against:

- a faster CPU implementation
- a CUDA implementation
- attention kernels

## Example: Full Basic Flow

```cpp
Tensor x({2, 3});

x.at({0, 0}) = 1.0f;
x.at({0, 1}) = 2.0f;
x.at({0, 2}) = 3.0f;
x.at({1, 0}) = 4.0f;
x.at({1, 1}) = 5.0f;
x.at({1, 2}) = 6.0f;

Tensor weights({3, 2});
weights.at({0, 0}) = 7.0f;
weights.at({0, 1}) = 8.0f;
weights.at({1, 0}) = 9.0f;
weights.at({1, 1}) = 10.0f;
weights.at({2, 0}) = 11.0f;
weights.at({2, 1}) = 12.0f;

Tensor y = x.matmul(weights);
```

The result is:

```text
[[58, 64],
 [139, 154]]
```

Because:

```text
y[0, 0] = 1*7 + 2*9 + 3*11 = 58
y[0, 1] = 1*8 + 2*10 + 3*12 = 64
y[1, 0] = 4*7 + 5*9 + 6*11 = 139
y[1, 1] = 4*8 + 5*10 + 6*12 = 154
```

## How This Supports TinyINR And INRFlow

The Tensor implementation is the lowest layer of TinyINR.

It does not perform INRFlow by itself. Instead, it makes the next modules possible:

### CoordinateBatch

`CoordinateBatch` can store:

```text
coordinates: [B, N, coordinate_dim]
values:      [B, N, value_dim]
```

For images:

```text
coordinates: [B, H * W, 2]
values:      [B, H * W, 3]
```

This depends on Tensor having stable shape metadata, safe indexing, and predictable contiguous storage.

### FourierEmbedding

`FourierEmbedding` will read coordinate tensors and produce embedding tensors:

```text
input:  [B, N, coordinate_dim]
output: [B, N, embedding_dim]
```

It will rely heavily on:

- `shape()`
- `data()`
- `at(...)`
- row-major memory layout

### Adaptive Coordinate Sampling

Later, TinyINR will sample useful coordinates instead of randomly choosing all points.

That needs Tensor support for:

- slicing or gathering coordinates
- preserving coordinate-value alignment
- comparing sampled values
- producing smaller coordinate batches

The current `row` method is a first small step toward slicing.

### Attention And Flow Matching

The plan includes latent-to-coordinate attention. Attention relies on matrix multiplication. The current `matmul` is only a baseline, but it gives a clear reference implementation.

Future CUDA kernels can be checked against this CPU behavior.

## Current Limitations

The implementation intentionally does not include:

- autograd
- CUDA memory
- arbitrary dtypes
- broadcasting
- batched matmul
- true zero-copy views
- slicing beyond `row`
- serialization
- deterministic random seeds

These are not bugs. They are scope boundaries for the first CPU-first Tensor core.

## Good Next Tests

The repo should eventually add tests for:

1. Constructor shape and zero initialization.
2. `numel()` for 1D, 2D, and 3D tensors.
3. `strides()` for `[2, 3]` and `[B, N, D]`.
4. `at(...)` reads and writes.
5. Out-of-bounds indexing throws.
6. Shape mismatch in `add` throws.
7. Shape mismatch in `multiply` throws.
8. Valid `reshape` preserves data order.
9. Invalid `reshape` throws.
10. `row` returns the expected slice.
11. 2D `matmul` gives known expected values.
12. Invalid `matmul` shapes throw.

## Reading The Code In Order

If you are learning the implementation from scratch, read the code in this order:

1. Private fields in `include/tensor.h`.
2. Constructor in `src/tensor.cpp`.
3. `compute_numel`.
4. `compute_strides`.
5. `offset`.
6. `at`.
7. `reshape` and `row`.
8. `add` and `multiply`.
9. `matmul`.
10. `randn`.

That order builds from memory ownership to layout to math operations.
