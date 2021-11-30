#pragma once

#include <torch/torch.h>

namespace fewbit {

torch::Tensor GeluCpu(torch::Tensor const &inputs, torch::Tensor const &bounds,
                      torch::Tensor const &levels);

std::tuple<torch::Tensor, torch::Tensor> Quantize(torch::Tensor const &inputs,
                                                  torch::Tensor const &bounds);

torch::Tensor QuantizeBackward(torch::Tensor const &grads,
                               torch::Tensor const &codes,
                               torch::Tensor const &levels);

} // namespace fewbit
