#include "gelu.h"

#include <cmath>

#include <torch/script.h>
#include <torch/torch.h>

#include <fewbit/cuda/codec.h>

namespace fewbit {

class GeluCudaFunction : public torch::autograd::Function<GeluCudaFunction> {
private:
    static auto constexpr kMaxBitWidth = 8;

public:
    static torch::Tensor forward(torch::autograd::AutogradContext *ctx,
                                 torch::Tensor const &inputs,
                                 torch::Tensor const &bounds,
                                 torch::Tensor const &levels) {
        TORCH_CHECK(levels.ndimension() == 1,
                    "Expected numer of dimensions doesn't equal to one.")
        TORCH_CHECK(bounds.ndimension() == 1,
                    "Expected numer of dimensions doesn't equal to one.")
        TORCH_CHECK(levels.numel() > 255,
                    "Expected number of levels exceeds 2^8 - 1 = 255.");
        TORCH_CHECK(levels.numel() == bounds.numel() + 1,
                    "Expected numel of `levels` and numel of `bounds` ",
                    "differs by one: ", levels.numel(), " - ", bounds.numel(),
                    " = ", levels.numel() - bounds.numel(), ".");

        auto nobits = static_cast<int32_t>(std::log2(bounds.numel()) + 0.5);
        auto nogroups = (inputs.numel() - 1) / kMaxBitWidth + 1;
        auto buffer_len = nobits * nogroups;
        auto buffer_opt = torch::TensorOptions()
                              .device(inputs.device())
                              .memory_format(torch::MemoryFormat::Contiguous)
                              .dtype(torch::kU8);
        auto buffer = torch::empty({buffer_len}, buffer_opt);
        auto output = torch::empty_like(inputs);
        ctx->save_for_backward({levels, buffer});
        Gelu(inputs.numel(), nobits, bounds.data_ptr<float>(),
             inputs.data_ptr<float>(), output.data_ptr<float>(),
             buffer.data_ptr<uint8_t>());
        return output;
    }

    static torch::autograd::variable_list
    backward(torch::autograd::AutogradContext *ctx,
             torch::autograd::variable_list grad_output) {
        auto state = ctx->get_saved_variables();
        auto nobits = static_cast<int32_t>(std::log2(state[0].numel()) + 0.5);
        auto grad_input = torch::empty_like(grad_output[0]);
        GeluBackward(grad_output[0].numel(), nobits, state[0].data_ptr<float>(),
                     state[1].data_ptr<uint8_t>(),
                     grad_output[0].data_ptr<float>(),
                     grad_input.data_ptr<float>());
        return {grad_input, torch::Tensor(), torch::Tensor()};
    }
};

torch::Tensor GeluCuda(torch::Tensor const &inputs, torch::Tensor const &bounds,
                       torch::Tensor const &levels) {
    return GeluCudaFunction::apply(inputs, bounds, levels);
}

} // namespace fewbit

TORCH_LIBRARY_IMPL(fewbit, AutogradCUDA, m) {
    m.impl("gelu", fewbit::GeluCuda);
}
