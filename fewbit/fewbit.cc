#include "fewbit.h"

#include <fewbit/codec.h>

namespace fewbit {

std::tuple<torch::Tensor, torch::Tensor> Quantize(torch::Tensor const &inputs,
                                                  torch::Tensor const &bounds) {
    auto outputs = torch::gelu(inputs);
    auto codes = torch::searchsorted(bounds, inputs, true);

    auto nobits = std::ceil(std::log2(bounds.numel()));
    auto length = static_cast<int64_t>(std::ceil(nobits * inputs.numel() / 8));
    auto buffer = torch::empty({length}, torch::kU8);

    Deflate(codes.data_ptr<int32_t>(),
            codes.data_ptr<int32_t>() + codes.numel(),
            buffer.data_ptr<uint8_t>(), static_cast<int32_t>(nobits));

    return {outputs, buffer};
}

torch::Tensor QuantizeBackward(torch::Tensor const &grads,
                               torch::Tensor const &buffer,
                               torch::Tensor const &levels) {
    auto nobits = std::ceil(std::log2(levels.numel()));
    auto codes = torch::empty_like(grads, torch::kInt32);
    Inflate(codes.data_ptr<int32_t>(),
            codes.data_ptr<int32_t>() + codes.numel(),
            buffer.data_ptr<uint8_t>(), static_cast<int32_t>(nobits));
    // TODO(@daskol): Parametrize compression codec routines with data
    // element type.
    auto factors = levels.index({codes.toType(torch::kInt64)});
    return factors * grads;
}

} // namespace fewbit
