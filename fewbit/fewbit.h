#pragma once

#include <torch/torch.h>

namespace fewbit {

std::tuple<torch::Tensor, torch::Tensor> Quantize(torch::Tensor const &inputs,
                                                  torch::Tensor const &bounds);

torch::Tensor QuantizeBackward(torch::Tensor const &grads,
                               torch::Tensor const &codes,
                               torch::Tensor const &levels);

} // namespace fewbit
