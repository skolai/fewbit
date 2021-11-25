#include "fewbit.h"

#include <fewbit/codec.h>

namespace fewbit {

std::tuple<torch::Tensor, torch::Tensor> Quantize(torch::Tensor const &inputs,
                                                  torch::Tensor const &bounds) {
#if __GNUC__ > 9
    auto outputs = torch::gelu(inputs);
#else
    auto outputs = torch::gelu(inputs, true);
#endif
    auto codes = torch::searchsorted(bounds, inputs, true);

    auto nobits = std::ceil(std::log2(bounds.numel()));
    auto length = static_cast<int64_t>(std::ceil(nobits * inputs.numel() / 8));
    auto buffer = torch::empty({length}, torch::kU8);

    Deflate(codes.data_ptr<int32_t>(),
            codes.data_ptr<int32_t>() + codes.numel(),
            buffer.data_ptr<uint8_t>(), static_cast<int32_t>(nobits));

#if __GNUC__ > 9
    return {outputs, buffer};
#else
    return std::make_tuple(outputs, buffer);
#endif
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

torch::Tensor Gelu::forward(torch::autograd::AutogradContext *ctx,
                            torch::Tensor const &inputs,
                            torch::Tensor const &bounds,
                            torch::Tensor const &levels) {
    auto [outputs, buffer] = Quantize(inputs, bounds);
    ctx->save_for_backward({buffer, levels});
    return outputs;
}

torch::autograd::variable_list
Gelu::backward(torch::autograd::AutogradContext *ctx,
               torch::autograd::variable_list grad_output) {
    auto vars = ctx->get_saved_variables();
    auto grad_inputs = QuantizeBackward(grad_output[0], vars[0], vars[1]);
    return {grad_inputs, torch::Tensor(), torch::Tensor()};
}

torch::Tensor GeluCpu(torch::Tensor const &inputs, torch::Tensor const &bounds,
                      torch::Tensor const &levels) {
    return Gelu::apply(inputs, bounds, levels);
}

} // namespace fewbit
