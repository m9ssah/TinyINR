#pragma once

#include "tensor.h"

Tensor FourierEmbedding {
  const Tensor &coords;
  const Tensor &frequencies;
}