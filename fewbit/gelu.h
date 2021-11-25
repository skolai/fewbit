#pragma once

#include <torch/torch.h>

namespace fewbit {

std::tuple<torch::Tensor, torch::Tensor> gelu_cuda(torch::Tensor const &inputs,
                                                   torch::Tensor const &bounds);

torch::Tensor gelu_backward_cuda(torch::Tensor const &grads,
                                 torch::Tensor const &buffer,
                                 torch::Tensor const &levels);

} // namespace fewbit
