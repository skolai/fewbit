#pragma once

#include <torch/torch.h>

namespace fewbit {

std::tuple<torch::Tensor, torch::Tensor> Quantize(torch::Tensor const &inputs,
                                                  torch::Tensor const &bounds);

torch::Tensor QuantizeBackward(torch::Tensor const &grads,
                               torch::Tensor const &codes,
                               torch::Tensor const &levels);

class Gelu : public torch::autograd::Function<Gelu> {
public:
    static torch::Tensor forward(torch::autograd::AutogradContext *ctx,
                                 torch::Tensor const &inputs,
                                 torch::Tensor const &bounds,
                                 torch::Tensor const &levels);

    static torch::autograd::variable_list
    backward(torch::autograd::AutogradContext *ctx,
             torch::autograd::variable_list grad_output);
};

torch::Tensor GeluCpu(torch::Tensor const &inputs, torch::Tensor const &bounds,
                      torch::Tensor const &levels);

} // namespace fewbit
