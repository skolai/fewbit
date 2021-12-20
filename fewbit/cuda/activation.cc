#include "activation.h"

#include <fewbit/cuda/codec.h>

namespace fewbit {

class HardshrinkCudaFunction
    : public torch::autograd::Function<HardshrinkCudaFunction> {
private:
    static auto constexpr kMaxBitWidth = 8;

public:
    static torch::Tensor forward(torch::autograd::AutogradContext *ctx,
                                 torch::Tensor const &inputs, double lambda) {
        auto buffer_len = (inputs.numel() - 1) / kMaxBitWidth + 1;
        auto buffer_opt = torch::TensorOptions()
                              .device(inputs.device())
                              .memory_format(torch::MemoryFormat::Contiguous)
                              .dtype(torch::kU8);
        auto buffer = torch::empty({buffer_len}, buffer_opt);
        ctx->mark_dirty({inputs});
        ctx->save_for_backward({buffer});
        Hardshrink(inputs.numel(), inputs.data_ptr<float>(),
                   inputs.data_ptr<float>(), buffer.data_ptr<uint8_t>(),
                   lambda);
        return inputs;
    }

    static torch::autograd::variable_list
    backward(torch::autograd::AutogradContext *ctx,
             torch::autograd::variable_list grad_output) {
        auto state = ctx->get_saved_variables();
        auto grad_input = torch::empty_like(grad_output[0]);
        HardshrinkBackward(grad_output[0].numel(), state[0].data_ptr<uint8_t>(),
                           grad_output[0].data_ptr<float>(),
                           grad_input.data_ptr<float>());
        return {grad_input, torch::Tensor()};
    }
};

torch::Tensor HardshrinkCuda(torch::Tensor const &inputs, double lambda) {
    return HardshrinkCudaFunction::apply(inputs, lambda);
}

class HardsigmoidCudaFunction
    : public torch::autograd::Function<HardsigmoidCudaFunction> {
private:
    static auto constexpr kMaxBitWidth = 8;

public:
    static torch::Tensor forward(torch::autograd::AutogradContext *ctx,
                                 torch::Tensor const &inputs) {
        auto buffer_len = (inputs.numel() - 1) / kMaxBitWidth + 1;
        auto buffer_opt = torch::TensorOptions()
                              .device(inputs.device())
                              .memory_format(torch::MemoryFormat::Contiguous)
                              .dtype(torch::kU8);
        auto buffer = torch::empty({buffer_len}, buffer_opt);
        ctx->mark_dirty({inputs});
        ctx->save_for_backward({buffer});
        Hardsigmoid(inputs.numel(), inputs.data_ptr<float>(),
                    inputs.data_ptr<float>(), buffer.data_ptr<uint8_t>());
        return inputs;
    }

    static torch::autograd::variable_list
    backward(torch::autograd::AutogradContext *ctx,
             torch::autograd::variable_list grad_output) {
        auto state = ctx->get_saved_variables();
        auto lambda = 0.5;
        auto grad_input = torch::empty_like(grad_output[0]);
        HardsigmoidBackward(
            grad_output[0].numel(), state[0].data_ptr<uint8_t>(),
            grad_output[0].data_ptr<float>(), grad_input.data_ptr<float>());
        return {grad_input, torch::Tensor()};
    }
};

torch::Tensor HardsigmoidCuda(torch::Tensor const &inputs) {
    return HardsigmoidCudaFunction::apply(inputs);
}

class HardtanhCudaFunction
    : public torch::autograd::Function<HardtanhCudaFunction> {
private:
    static auto constexpr kMaxBitWidth = 8;

public:
    static torch::Tensor forward(torch::autograd::AutogradContext *ctx,
                                 torch::Tensor const &inputs, double min_val,
                                 double max_val) {
        auto buffer_len = (inputs.numel() - 1) / kMaxBitWidth + 1;
        auto buffer_opt = torch::TensorOptions()
                              .device(inputs.device())
                              .memory_format(torch::MemoryFormat::Contiguous)
                              .dtype(torch::kU8);
        auto buffer = torch::empty({buffer_len}, buffer_opt);
        ctx->mark_dirty({inputs});
        ctx->save_for_backward({buffer});
        Hardtanh(inputs.numel(), inputs.data_ptr<float>(),
                 inputs.data_ptr<float>(), buffer.data_ptr<uint8_t>(), min_val,
                 max_val);
        return inputs;
    }

    static torch::autograd::variable_list
    backward(torch::autograd::AutogradContext *ctx,
             torch::autograd::variable_list grad_output) {
        auto state = ctx->get_saved_variables();
        auto lambda = 0.5;
        auto grad_input = torch::empty_like(grad_output[0]);
        HardtanhBackward(grad_output[0].numel(), state[0].data_ptr<uint8_t>(),
                         grad_output[0].data_ptr<float>(),
                         grad_input.data_ptr<float>());
        return {grad_input, torch::Tensor()};
    }
};

torch::Tensor HardtanhCuda(torch::Tensor const &inputs, double min_val,
                           double max_val) {
    return HardtanhCudaFunction::apply(inputs, min_val, max_val);
}

class LeakyReluCudaFunction
    : public torch::autograd::Function<LeakyReluCudaFunction> {
private:
    static auto constexpr kMaxBitWidth = 8;

public:
    static torch::Tensor forward(torch::autograd::AutogradContext *ctx,
                                 torch::Tensor const &inputs,
                                 double negative_slope) {
        auto buffer_len = (inputs.numel() - 1) / kMaxBitWidth + 1;
        auto buffer_opt = torch::TensorOptions()
                              .device(inputs.device())
                              .memory_format(torch::MemoryFormat::Contiguous)
                              .dtype(torch::kU8);
        auto buffer = torch::empty({buffer_len}, buffer_opt);
        auto params = torch::tensor({negative_slope});
        ctx->mark_dirty({inputs});
        ctx->save_for_backward({buffer, params});
        LeakyRelu(inputs.numel(), inputs.data_ptr<float>(),
                  inputs.data_ptr<float>(), buffer.data_ptr<uint8_t>(),
                  negative_slope);
        return inputs;
    }

    static torch::autograd::variable_list
    backward(torch::autograd::AutogradContext *ctx,
             torch::autograd::variable_list grad_output) {
        auto state = ctx->get_saved_variables();
        auto negative_slope = state[1][0].item<double>();
        auto grad_input = torch::empty_like(grad_output[0]);
        LeakyReluBackward(grad_output[0].numel(), state[0].data_ptr<uint8_t>(),
                          grad_output[0].data_ptr<float>(),
                          grad_input.data_ptr<float>(), negative_slope);
        return {grad_input, torch::Tensor()};
    }
};

torch::Tensor LeakyReluCuda(torch::Tensor const &inputs,
                            double negative_slope) {
    return LeakyReluCudaFunction::apply(inputs, negative_slope);
}

class ReluCudaFunction : public torch::autograd::Function<ReluCudaFunction> {
private:
    static auto constexpr kMaxBitWidth = 8;

public:
    static torch::Tensor forward(torch::autograd::AutogradContext *ctx,
                                 torch::Tensor const &inputs) {
        auto buffer_len = (inputs.numel() - 1) / kMaxBitWidth + 1;
        auto buffer_opt = torch::TensorOptions()
                              .device(inputs.device())
                              .memory_format(torch::MemoryFormat::Contiguous)
                              .dtype(torch::kU8);
        auto buffer = torch::empty({buffer_len}, buffer_opt);
        ctx->mark_dirty({inputs});
        ctx->save_for_backward({buffer});
        Relu(inputs.numel(), inputs.data_ptr<float>(), inputs.data_ptr<float>(),
             buffer.data_ptr<uint8_t>());
        return inputs;
    }

    static torch::autograd::variable_list
    backward(torch::autograd::AutogradContext *ctx,
             torch::autograd::variable_list grad_output) {
        auto state = ctx->get_saved_variables();
        auto grad_input = torch::empty_like(grad_output[0]);
        ReluBackward(grad_output[0].numel(), state[0].data_ptr<uint8_t>(),
                     grad_output[0].data_ptr<float>(),
                     grad_input.data_ptr<float>());
        return {grad_input, torch::Tensor()};
    }
};

torch::Tensor ReluCuda(torch::Tensor const &inputs) {
    return ReluCudaFunction::apply(inputs);
}

class Relu6CudaFunction : public torch::autograd::Function<Relu6CudaFunction> {
private:
    static auto constexpr kMaxBitWidth = 8;

public:
    static torch::Tensor forward(torch::autograd::AutogradContext *ctx,
                                 torch::Tensor const &inputs) {
        auto buffer_len = (inputs.numel() - 1) / kMaxBitWidth + 1;
        auto buffer_opt = torch::TensorOptions()
                              .device(inputs.device())
                              .memory_format(torch::MemoryFormat::Contiguous)
                              .dtype(torch::kU8);
        auto buffer = torch::empty({buffer_len}, buffer_opt);
        ctx->mark_dirty({inputs});
        ctx->save_for_backward({buffer});
        Relu6(inputs.numel(), inputs.data_ptr<float>(),
              inputs.data_ptr<float>(), buffer.data_ptr<uint8_t>());
        return inputs;
    }

    static torch::autograd::variable_list
    backward(torch::autograd::AutogradContext *ctx,
             torch::autograd::variable_list grad_output) {
        auto state = ctx->get_saved_variables();
        auto grad_input = torch::empty_like(grad_output[0]);
        Relu6Backward(grad_output[0].numel(), state[0].data_ptr<uint8_t>(),
                      grad_output[0].data_ptr<float>(),
                      grad_input.data_ptr<float>());
        return {grad_input, torch::Tensor()};
    }
};

torch::Tensor Relu6Cuda(torch::Tensor const &inputs) {
    return Relu6CudaFunction::apply(inputs);
}

class SoftshrinkCudaFunction
    : public torch::autograd::Function<SoftshrinkCudaFunction> {
private:
    static auto constexpr kMaxBitWidth = 8;

public:
    static torch::Tensor forward(torch::autograd::AutogradContext *ctx,
                                 torch::Tensor const &inputs, double lambda) {
        auto buffer_len = (inputs.numel() - 1) / kMaxBitWidth + 1;
        auto buffer_opt = torch::TensorOptions()
                              .device(inputs.device())
                              .memory_format(torch::MemoryFormat::Contiguous)
                              .dtype(torch::kU8);
        auto buffer = torch::empty({buffer_len}, buffer_opt);
        ctx->mark_dirty({inputs});
        ctx->save_for_backward({buffer});
        Softshrink(inputs.numel(), inputs.data_ptr<float>(),
                   inputs.data_ptr<float>(), buffer.data_ptr<uint8_t>(),
                   lambda);
        return inputs;
    }

    static torch::autograd::variable_list
    backward(torch::autograd::AutogradContext *ctx,
             torch::autograd::variable_list grad_output) {
        auto state = ctx->get_saved_variables();
        auto grad_input = torch::empty_like(grad_output[0]);
        SoftshrinkBackward(grad_output[0].numel(), state[0].data_ptr<uint8_t>(),
                           grad_output[0].data_ptr<float>(),
                           grad_input.data_ptr<float>());
        return {grad_input, torch::Tensor()};
    }
};

torch::Tensor SoftshrinkCuda(torch::Tensor const &inputs, double lambda) {
    return SoftshrinkCudaFunction::apply(inputs, lambda);
}

class ThresholdCudaFunction
    : public torch::autograd::Function<ThresholdCudaFunction> {
private:
    static auto constexpr kMaxBitWidth = 8;

public:
    static torch::Tensor forward(torch::autograd::AutogradContext *ctx,
                                 torch::Tensor const &inputs, double threshold,
                                 double value) {
        auto buffer_len = (inputs.numel() - 1) / kMaxBitWidth + 1;
        auto buffer_opt = torch::TensorOptions()
                              .device(inputs.device())
                              .memory_format(torch::MemoryFormat::Contiguous)
                              .dtype(torch::kU8);
        auto buffer = torch::empty({buffer_len}, buffer_opt);
        ctx->mark_dirty({inputs});
        ctx->save_for_backward({buffer});
        Threshold(inputs.numel(), inputs.data_ptr<float>(),
                  inputs.data_ptr<float>(), buffer.data_ptr<uint8_t>(),
                  threshold, value);
        return inputs;
    }

    static torch::autograd::variable_list
    backward(torch::autograd::AutogradContext *ctx,
             torch::autograd::variable_list grad_output) {
        auto state = ctx->get_saved_variables();
        auto grad_input = torch::empty_like(grad_output[0]);
        ThresholdBackward(grad_output[0].numel(), state[0].data_ptr<uint8_t>(),
                          grad_output[0].data_ptr<float>(),
                          grad_input.data_ptr<float>());
        return {grad_input, torch::Tensor()};
    }
};

torch::Tensor ThresholdCuda(torch::Tensor const &inputs, double threshold,
                            double value) {
    return ThresholdCudaFunction::apply(inputs, threshold, value);
}

TORCH_LIBRARY_IMPL(fewbit, AutogradCUDA, m) {
    m.impl("hardshrink", fewbit::HardshrinkCuda);
    m.impl("hardsigmoid", fewbit::HardsigmoidCuda);
    m.impl("hardtanh", fewbit::HardtanhCuda);
    m.impl("leaky_relu", fewbit::LeakyReluCuda);
    m.impl("relu", fewbit::ReluCuda);
    m.impl("relu6", fewbit::Relu6Cuda);
    m.impl("softshrink", fewbit::SoftshrinkCuda);
    m.impl("threshold", fewbit::ThresholdCuda);
}

} // namespace fewbit
