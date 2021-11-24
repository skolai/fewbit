#include "fewbit.h"

namespace fewbit {

std::tuple<torch::Tensor, torch::Tensor> Quantize(torch::Tensor const &inputs,
                                                  torch::Tensor const &bounds) {
    auto outputs = torch::gelu(inputs);
    auto codes = torch::searchsorted(bounds, inputs, false);
    return {outputs, codes};
}

torch::Tensor QuantizeBackward(torch::Tensor const &grads,
                               torch::Tensor const &codes,
                               torch::Tensor const &levels) {
    auto mult = levels.index({codes});
    return mult * grads;
}

} // namespace fewbit
