#pragma once

#include <cstdint> 
#include <vector>

class Tensor {
public:
    explicit Tensor(const std::vector<int64_t>& shape);

    static Tensor zeros(const std::vector<int64_t>& shape);
    static Tensor randn(const std::vector<int64_t>& shape, float mean = 0.0f, float stddev = 1.0f);

    float* data();
    const float* data() const;

    const std::vector<int64_t>& shape() const;
    const std::vector<int64_t>& strides() const;

    int64_t ndim() const;
    int64_t numel() const;

    float& at(const std::vector<int64_t>& indices);
    const float& at(const std::vector<int64_t>& indices) const;

    Tensor reshape(const std::vector<int64_t>& new_shape) const;
    Tensor row(int64_t row_index) const;

    Tensor add(const Tensor& other) const;
    Tensor multiply(const Tensor& other) const;
    Tensor matmul(const Tensor& other) const;

private:
    std::vector<float> data_;
    std::vector<int64_t> shape_;
    std::vector<int64_t> strides_;

    static int64_t compute_numel(const std::vector<int64_t>& shape);
    static std::vector<int64_t> compute_strides(const std::vector<int64_t>& shape);

    int64_t offset(const std::vector<int64_t>& indices) const;
};