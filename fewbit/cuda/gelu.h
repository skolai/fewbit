#pragma once

#include <torch/torch.h>

namespace fewbit {

torch::Tensor GeluCuda(torch::Tensor const &inputs, torch::Tensor const &bounds,
                       torch::Tensor const &levels);

} // namespace fewbit
