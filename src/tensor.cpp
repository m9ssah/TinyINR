#include "tensor.h"

#include <algorithm>
#include <limits>
#include <random>
#include <stdexcept>

Tensor::Tensor(const std::vector<int64_t> &shape)
    : data_(compute_numel(shape), 0.0f), shape_(shape),
      strides_(compute_strides(shape)) {}

Tensor Tensor::zeros(const std::vector<int64_t> &shape) {
  return Tensor(shape);
}

Tensor Tensor::randn(const std::vector<int64_t> &shape, float mean,
                     float stddev) {
  if (stddev < 0.0f) {
    throw std::invalid_argument("randn stddev must be non-negative");
  }

  Tensor result(shape);
  std::random_device seed;
  std::mt19937 generator(seed());
  std::normal_distribution<float> distribution(mean, stddev);

  for (int64_t i = 0; i < result.numel(); ++i) {
    result.data_[static_cast<size_t>(i)] = distribution(generator);
  }

  return result;
}

float *Tensor::data() { return data_.data(); }

const float *Tensor::data() const { return data_.data(); }

const std::vector<int64_t> &Tensor::shape() const { return shape_; }

const std::vector<int64_t> &Tensor::strides() const { return strides_; }

int64_t Tensor::ndim() const { return static_cast<int64_t>(shape_.size()); }

int64_t Tensor::numel() const { return static_cast<int64_t>(data_.size()); }

float &Tensor::at(const std::vector<int64_t> &indices) {
  return data_[static_cast<size_t>(offset(indices))];
}

const float &Tensor::at(const std::vector<int64_t> &indices) const {
  return data_[static_cast<size_t>(offset(indices))];
}

Tensor Tensor::reshape(const std::vector<int64_t> &new_shape) const {
  if (compute_numel(new_shape) != numel()) {
    throw std::invalid_argument("reshape cannot change the number of elements");
  }

  Tensor result(new_shape);
  result.data_ = data_;
  return result;
}

Tensor Tensor::row(int64_t row_index) const {
  if (shape_.size() < 2) {
    throw std::invalid_argument(
        "row requires a tensor with at least 2 dimensions");
  }
  if (row_index < 0 || row_index >= shape_[0]) {
    throw std::out_of_range("row index out of bounds");
  }

  std::vector<int64_t> row_shape(shape_.begin() + 1, shape_.end());
  Tensor result(row_shape);

  const int64_t row_size = result.numel();
  const int64_t source_start = row_index * row_size;

  std::copy(data_.begin() + source_start,
            data_.begin() + source_start + row_size, result.data_.begin());

  return result;
}

Tensor Tensor::add(const Tensor &other) const {
  if (shape_ != other.shape_) {
    throw std::invalid_argument("add requires tensors with the same shape");
  }

  Tensor result(shape_);
  for (int64_t i = 0; i < numel(); ++i) {
    const size_t index = static_cast<size_t>(i);
    result.data_[index] = data_[index] + other.data_[index];
  }

  return result;
}

Tensor Tensor::multiply(const Tensor &other) const {
  if (shape_ != other.shape_) {
    throw std::invalid_argument(
        "multiply requires tensors with the same shape");
  }

  Tensor result(shape_);
  for (int64_t i = 0; i < numel(); ++i) {
    const size_t index = static_cast<size_t>(i);
    result.data_[index] = data_[index] * other.data_[index];
  }

  return result;
}

Tensor Tensor::matmul(const Tensor &other) const {
  if (shape_.size() != 2 || other.shape_.size() != 2) {
    throw std::invalid_argument("matmul currently supports only 2D tensors");
  }

  const int64_t rows = shape_[0];
  const int64_t inner = shape_[1];
  const int64_t other_inner = other.shape_[0];
  const int64_t cols = other.shape_[1];

  if (inner != other_inner) {
    throw std::invalid_argument("matmul inner dimensions must match");
  }

  Tensor result({rows, cols});

  for (int64_t i = 0; i < rows; ++i) {
    for (int64_t j = 0; j < cols; ++j) {
      float sum = 0.0f;
      for (int64_t k = 0; k < inner; ++k) {
        sum += at({i, k}) * other.at({k, j});
      }
      result.at({i, j}) = sum;
    }
  }

  return result;
}

int64_t Tensor::compute_numel(const std::vector<int64_t> &shape) {
  if (shape.empty()) {
    throw std::invalid_argument(
        "tensor shape must contain at least one dimension");
  }

  int64_t total = 1;
  for (int64_t dim : shape) {
    if (dim <= 0) {
      throw std::invalid_argument("tensor dimensions must be positive");
    }
    if (total > std::numeric_limits<int64_t>::max() / dim) {
      throw std::overflow_error("tensor shape is too large");
    }
    total *= dim;
  }

  return total;
}

std::vector<int64_t>
Tensor::compute_strides(const std::vector<int64_t> &shape) {
  std::vector<int64_t> strides(shape.size());
  int64_t stride = 1;

  for (int64_t i = static_cast<int64_t>(shape.size()) - 1; i >= 0; --i) {
    strides[static_cast<size_t>(i)] = stride;
    stride *= shape[static_cast<size_t>(i)];
  }

  return strides;
}

int64_t Tensor::offset(const std::vector<int64_t> &indices) const {
  if (indices.size() != shape_.size()) {
    throw std::invalid_argument("incorrect number of tensor indices");
  }

  int64_t flat_index = 0;
  for (size_t i = 0; i < indices.size(); ++i) {
    if (indices[i] < 0 || indices[i] >= shape_[i]) {
      throw std::out_of_range("tensor index out of bounds");
    }
    flat_index += indices[i] * strides_[i];
  }

  return flat_index;
}
