#include "gelu.h"

#include <torch/script.h>

namespace fewbit {

class GeluCuda : public torch::autograd::Function<GeluCuda> {
public:
    static torch::Tensor forward(torch::autograd::AutogradContext *ctx,
                                 torch::Tensor const &inputs,
                                 torch::Tensor const &bounds,
                                 torch::Tensor const &levels) {
        ctx->save_for_backward({inputs});
        return 2 * torch::ones_like(inputs);
    }

    static torch::autograd::variable_list
    backward(torch::autograd::AutogradContext *ctx,
             torch::autograd::variable_list grad_output) {
        auto state = ctx->get_saved_variables();
        return {3 * torch::ones_like(state[0]), torch::Tensor(),
                torch::Tensor()};
    }
};

torch::Tensor GeluGpu(torch::Tensor const &inputs, torch::Tensor const &bounds,
                      torch::Tensor const &levels) {
    return GeluCuda::apply(inputs, bounds, levels);
}

} // namespace fewbit

TORCH_LIBRARY_IMPL(fewbit, AutogradCUDA, m) {
    m.impl("gelu", fewbit::GeluGpu);
}
