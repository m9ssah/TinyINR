#pragma once

#include <cstdint>
#include <vector>

/**
 * A small CPU-only float32 tensor for TinyINR.
 *
 * Tensor owns a flat std::vector<float> buffer and interprets it with shape and
 * stride metadata. Storage is contiguous and row-major.
 *
 * 
 */
class Tensor {
public:
    /**
     * Create a zero-initialized tensor with the given shape.
     *
     * All dimensions must be positive. The number of elements is the product of
     * the dimensions.
     */
    explicit Tensor(const std::vector<int64_t>& shape);

    /**
     * Create a tensor filled with zeros.
     */
    static Tensor zeros(const std::vector<int64_t>& shape);

    /**
     * Create a tensor filled with normally distributed random values.
     *
     * mean controls the center of the distribution and stddev controls its
     * spread. stddev must be non-negative.
     */
    static Tensor randn(const std::vector<int64_t>& shape, float mean = 0.0f, float stddev = 1.0f);

    /**
     * Return a mutable pointer to the contiguous float buffer.
     */
    float* data();

    /**
     * Return a read-only pointer to the contiguous float buffer.
     */
    const float* data() const;

    /**
     * Return the tensor shape, such as [batch, points, channels].
     */
    const std::vector<int64_t>& shape() const;

    /**
     * Return row-major strides used to map multidimensional indices to offsets.
     */
    const std::vector<int64_t>& strides() const;

    /**
     * Return the number of tensor dimensions.
     */
    int64_t ndim() const;

    /**
     * Return the total number of elements in the tensor.
     */
    int64_t numel() const;

    /**
     * Return a mutable reference to one element using checked indexing.
     *
     * Example for shape [2, 3]:
     *   x.at({1, 2}) accesses row 1, column 2.
     */
    float& at(const std::vector<int64_t>& indices);

    /**
     * Return a read-only reference to one element using checked indexing.
     */
    const float& at(const std::vector<int64_t>& indices) const;

    /**
     * Return a copy of this tensor with a different shape.
     *
     * The new shape must contain the same number of elements. This first
     * implementation copies data rather than creating a view, which keeps memory
     * ownership simple and safe.
     */
    Tensor reshape(const std::vector<int64_t>& new_shape) const;

    /**
     * Return a copy of one slice along the first dimension.
     *
     * For a [rows, cols] tensor, row(i) returns shape [cols]. For a
     * [B, N, D] tensor, row(i) returns shape [N, D].
     */
    Tensor row(int64_t row_index) const;

    /**
     * Elementwise addition with another tensor of the exact same shape.
     */
    Tensor add(const Tensor& other) const;

    /**
     * Elementwise multiplication with another tensor of the exact same shape.
     */
    Tensor multiply(const Tensor& other) const;

    /**
     * Basic 2D matrix multiplication.
     *
     * If this tensor has shape [M, K] and other has shape [K, N], the result has
     * shape [M, N].
     */
    Tensor matmul(const Tensor& other) const;

private:
    std::vector<float> data_;
    std::vector<int64_t> shape_;
    std::vector<int64_t> strides_;

    static int64_t compute_numel(const std::vector<int64_t>& shape);
    static std::vector<int64_t> compute_strides(const std::vector<int64_t>& shape);

    int64_t offset(const std::vector<int64_t>& indices) const;
};
