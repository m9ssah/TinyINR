#include "model/mlp.h"

#include <cmath>
#include <random>
#include <stdexcept>
#include <string>

namespace {

void require2d(const Tensor &tensor, const char *name) {
  if (tensor.ndim() != 2) {
    throw std::invalid_argument(std::string(name) + " must be 2D");
  }
}

Tensor flattenModelInput(const Tensor &input) {
  if (input.ndim() == 2) {
    return input;
  }
  if (input.ndim() != 3) {
    throw std::invalid_argument("MLP input must have shape [R, D] or [B, N, D]");
  }

  return input.reshape({input.shape()[0] * input.shape()[1], input.shape()[2]});
}

Tensor flattenModelOutputGradient(const Tensor &gradient,
                                  const std::vector<int64_t> &output_shape) {
  if (gradient.shape() == output_shape && gradient.ndim() == 3) {
    return gradient.reshape(
        {gradient.shape()[0] * gradient.shape()[1], gradient.shape()[2]});
  }
  if (gradient.ndim() == 2) {
    return gradient;
  }
  throw std::invalid_argument("output gradient shape does not match MLP output");
}

Tensor linearForward(const Tensor &input, const LinearLayer &layer) {
  require2d(input, "linear input");

  const int64_t rows = input.shape()[0];
  const int64_t in_dim = input.shape()[1];
  const int64_t out_dim = layer.weight.shape()[0];

  if (layer.weight.shape()[1] != in_dim || layer.bias.shape()[0] != out_dim) {
    throw std::invalid_argument("linear layer dimensions do not match input");
  }

  Tensor output({rows, out_dim});
  for (int64_t row = 0; row < rows; ++row) {
    for (int64_t out = 0; out < out_dim; ++out) {
      float sum = layer.bias.at({out});
      for (int64_t in = 0; in < in_dim; ++in) {
        sum += layer.weight.at({out, in}) * input.at({row, in});
      }
      output.at({row, out}) = sum;
    }
  }

  return output;
}

Tensor linearBackwardInput(const Tensor &upstream, const Tensor &weight) {
  require2d(upstream, "linear upstream");

  const int64_t rows = upstream.shape()[0];
  const int64_t out_dim = upstream.shape()[1];
  const int64_t in_dim = weight.shape()[1];

  if (weight.shape()[0] != out_dim) {
    throw std::invalid_argument("linear backward weight shape mismatch");
  }

  Tensor grad_input({rows, in_dim});
  for (int64_t row = 0; row < rows; ++row) {
    for (int64_t in = 0; in < in_dim; ++in) {
      float sum = 0.0f;
      for (int64_t out = 0; out < out_dim; ++out) {
        sum += upstream.at({row, out}) * weight.at({out, in});
      }
      grad_input.at({row, in}) = sum;
    }
  }
  return grad_input;
}

void accumulateLinearGradients(LinearLayer &layer, const Tensor &input,
                               const Tensor &upstream) {
  require2d(input, "linear gradient input");
  require2d(upstream, "linear gradient upstream");

  const int64_t rows = input.shape()[0];
  const int64_t in_dim = input.shape()[1];
  const int64_t out_dim = upstream.shape()[1];

  if (layer.weight.shape() != std::vector<int64_t>({out_dim, in_dim})) {
    throw std::invalid_argument("linear gradient layer shape mismatch");
  }

  for (int64_t out = 0; out < out_dim; ++out) {
    float bias_sum = 0.0f;
    for (int64_t in = 0; in < in_dim; ++in) {
      float weight_sum = 0.0f;
      for (int64_t row = 0; row < rows; ++row) {
        weight_sum += upstream.at({row, out}) * input.at({row, in});
      }
      layer.grad_weight.at({out, in}) += weight_sum;
    }

    for (int64_t row = 0; row < rows; ++row) {
      bias_sum += upstream.at({row, out});
    }
    layer.grad_bias.at({out}) += bias_sum;
  }
}

float sigmoid(float value) { return 1.0f / (1.0f + std::exp(-value)); }

} // namespace

LinearLayer::LinearLayer(int64_t in_dim, int64_t out_dim)
    : weight({out_dim, in_dim}), bias({out_dim}),
      grad_weight({out_dim, in_dim}), grad_bias({out_dim}) {}

Mlp::Mlp(const MlpConfig &config) : config(config) {
  if (config.input_dim <= 0 || config.hidden_width <= 0 ||
      config.output_dim <= 0) {
    throw std::invalid_argument("MLP dimensions must be positive");
  }
  if (config.hidden_layer_count != 3) {
    throw std::invalid_argument("baseline MLP requires 3 hidden layers");
  }

  layers.emplace_back(config.input_dim, config.hidden_width);
  layers.emplace_back(config.hidden_width, config.hidden_width);
  layers.emplace_back(config.hidden_width, config.hidden_width);
  layers.emplace_back(config.hidden_width, config.output_dim);
}

MlpConfig makeBaselineMlpConfig(int64_t input_dim, int64_t output_dim) {
  MlpConfig config;
  config.input_dim = input_dim;
  config.hidden_width = 256;
  config.output_dim = output_dim;
  config.hidden_layer_count = 3;
  config.activation = ActivationKind::SiLU;
  return config;
}

Mlp createMlp(const MlpConfig &config, uint32_t seed) {
  Mlp model(config);
  std::mt19937 generator(seed);

  for (size_t layer_index = 0; layer_index < model.layers.size();
       ++layer_index) {
    LinearLayer &layer = model.layers[layer_index];
    const float fan_in = static_cast<float>(layer.weight.shape()[1]);
    const float stddev = std::sqrt(2.0f / fan_in);
    std::normal_distribution<float> distribution(0.0f, stddev);

    for (int64_t i = 0; i < layer.weight.numel(); ++i) {
      layer.weight.data()[static_cast<size_t>(i)] = distribution(generator);
    }
    for (int64_t i = 0; i < layer.bias.numel(); ++i) {
      layer.bias.data()[static_cast<size_t>(i)] = 0.0f;
    }
  }

  return model;
}

Tensor silu(const Tensor &input) {
  Tensor output(input.shape());
  for (int64_t i = 0; i < input.numel(); ++i) {
    const float value = input.data()[static_cast<size_t>(i)];
    output.data()[static_cast<size_t>(i)] = value * sigmoid(value);
  }
  return output;
}

Tensor siluBackward(const Tensor &pre_activation, const Tensor &upstream) {
  if (pre_activation.shape() != upstream.shape()) {
    throw std::invalid_argument(
        "SiLU backward requires matching tensor shapes");
  }

  Tensor grad(pre_activation.shape());
  for (int64_t i = 0; i < pre_activation.numel(); ++i) {
    const float value = pre_activation.data()[static_cast<size_t>(i)];
    const float sig = sigmoid(value);
    const float derivative = sig + value * sig * (1.0f - sig);
    grad.data()[static_cast<size_t>(i)] =
        upstream.data()[static_cast<size_t>(i)] * derivative;
  }
  return grad;
}

MlpForwardResult mlpForward(const Mlp &model, const Tensor &input) {
  Tensor x = flattenModelInput(input);
  if (x.shape()[1] != model.config.input_dim) {
    throw std::invalid_argument("MLP input dimension mismatch");
  }

  std::vector<Tensor> pre_activations;
  std::vector<Tensor> activations;
  Tensor current = x;

  for (size_t i = 0; i + 1 < model.layers.size(); ++i) {
    Tensor z = linearForward(current, model.layers[i]);
    Tensor a = silu(z);
    pre_activations.push_back(z);
    activations.push_back(a);
    current = activations.back();
  }

  Tensor y = linearForward(current, model.layers.back());
  std::vector<int64_t> output_shape;
  Tensor output = y;
  if (input.ndim() == 3) {
    output_shape = {input.shape()[0], input.shape()[1], model.config.output_dim};
    output = y.reshape(output_shape);
  } else {
    output_shape = y.shape();
  }

  MlpForwardCache cache{x, pre_activations, activations, input.shape(),
                        output_shape};
  return MlpForwardResult{output, cache};
}

Tensor mlpBackward(Mlp &model, const MlpForwardCache &cache,
                   const Tensor &output_gradient) {
  if (cache.pre_activations.size() != 3 || cache.activations.size() != 3) {
    throw std::invalid_argument("MLP cache is missing hidden activations");
  }

  Tensor upstream =
      flattenModelOutputGradient(output_gradient, cache.output_shape);

  const Tensor &a2 = cache.activations[2];
  accumulateLinearGradients(model.layers[3], a2, upstream);
  upstream = linearBackwardInput(upstream, model.layers[3].weight);

  upstream = siluBackward(cache.pre_activations[2], upstream);
  const Tensor &a1 = cache.activations[1];
  accumulateLinearGradients(model.layers[2], a1, upstream);
  upstream = linearBackwardInput(upstream, model.layers[2].weight);

  upstream = siluBackward(cache.pre_activations[1], upstream);
  const Tensor &a0 = cache.activations[0];
  accumulateLinearGradients(model.layers[1], a0, upstream);
  upstream = linearBackwardInput(upstream, model.layers[1].weight);

  upstream = siluBackward(cache.pre_activations[0], upstream);
  accumulateLinearGradients(model.layers[0], cache.input, upstream);
  Tensor grad_input = linearBackwardInput(upstream, model.layers[0].weight);

  if (cache.input_shape.size() == 3) {
    return grad_input.reshape(cache.input_shape);
  }

  return grad_input;
}

void zeroGrad(Mlp &model) {
  for (size_t layer_index = 0; layer_index < model.layers.size();
       ++layer_index) {
    LinearLayer &layer = model.layers[layer_index];
    for (int64_t i = 0; i < layer.grad_weight.numel(); ++i) {
      layer.grad_weight.data()[static_cast<size_t>(i)] = 0.0f;
    }
    for (int64_t i = 0; i < layer.grad_bias.numel(); ++i) {
      layer.grad_bias.data()[static_cast<size_t>(i)] = 0.0f;
    }
  }
}

void sgdStep(Mlp &model, float learning_rate) {
  if (learning_rate <= 0.0f) {
    throw std::invalid_argument("learning_rate must be positive");
  }

  for (size_t layer_index = 0; layer_index < model.layers.size();
       ++layer_index) {
    LinearLayer &layer = model.layers[layer_index];
    for (int64_t i = 0; i < layer.weight.numel(); ++i) {
      const size_t index = static_cast<size_t>(i);
      layer.weight.data()[index] -= learning_rate * layer.grad_weight.data()[index];
    }
    for (int64_t i = 0; i < layer.bias.numel(); ++i) {
      const size_t index = static_cast<size_t>(i);
      layer.bias.data()[index] -= learning_rate * layer.grad_bias.data()[index];
    }
  }
}
