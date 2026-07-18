#pragma once

#include "../tensor.h"

struct FourierEmbedding {
  const Tensor &coords;
  const Tensor &frequencies;
};