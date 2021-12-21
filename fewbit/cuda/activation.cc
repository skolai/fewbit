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
        return {grad_input, torch::Tensor(), torch::Tensor()};
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
        return {grad_input};
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
        return {grad_input};
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
        return {grad_input, torch::Tensor(), torch::Tensor()};
    }
};

torch::Tensor ThresholdCuda(torch::Tensor const &inputs, double threshold,
                            double value) {
    return ThresholdCudaFunction::apply(inputs, threshold, value);
}

/**
 * Class ContinousCudaFunction is parametrized with some type T, which has
 * static method Invoke. The only purpose of static method of type T is to
 * dispatch call to actual function which corresponds to type T.
 */
template <typename T>
class ContinousCudaFunction
    : public torch::autograd::Function<ContinousCudaFunction<T>> {
private:
    static auto constexpr kMaxBitWidth = 8;

public:
    template <typename... Args>
    static torch::Tensor forward(torch::autograd::AutogradContext *ctx,
                                 torch::Tensor const &inputs,
                                 torch::Tensor const &bounds,
                                 torch::Tensor const &levels, Args &&...args) {
        auto nobits = static_cast<int32_t>(std::log2(bounds.numel()) + 0.5);
        auto nogroups = (inputs.numel() - 1) / kMaxBitWidth + 1;
        auto buffer_len = nobits * nogroups;
        auto buffer_opt = torch::TensorOptions()
                              .device(inputs.device())
                              .memory_format(torch::MemoryFormat::Contiguous)
                              .dtype(torch::kU8);
        auto buffer = torch::empty({buffer_len}, buffer_opt);
        ctx->mark_dirty({inputs});
        ctx->save_for_backward({buffer, levels});
        T::Invoke(inputs.numel(), inputs.data_ptr<float>(),
                  inputs.data_ptr<float>(), buffer.data_ptr<uint8_t>(), nobits,
                  bounds.data_ptr<float>(), std::forward<Args>(args)...);
        return inputs;
    }

    static torch::autograd::variable_list
    backward(torch::autograd::AutogradContext *ctx,
             torch::autograd::variable_list grad_output) {
        auto state = ctx->get_saved_variables();
        auto nobits = static_cast<int32_t>(std::log2(state[0].numel()) + 0.5);
        // All continous activation functions have the same pattern in their
        // signatures: all auxiliary parameters are appened to the end.
        auto constexpr nograds_base = 3;
        auto constexpr nograds = T::kNoArgs - nograds_base;
        torch::autograd::variable_list grad_input(nograds);
        grad_input[0] = torch::empty_like(grad_output[0]);
        StepwiseBackward(grad_output[0].numel(), state[0].data_ptr<uint8_t>(),
                         grad_output[0].data_ptr<float>(),
                         grad_input[0].data_ptr<float>(), nobits,
                         state[1].data_ptr<float>());
        return grad_input;
    }
};

/**
 * This macro definition is used to bind a specific CUDA kernel launcher to a
 * specific type. This is how we dispatch functions at compile-time. We assume
 * that there is only one kernel launcher with that name (no overloads).
 */

template <typename Ret, typename... Args>
constexpr auto CountFunctionArgs(Ret(Args...)) {
    return sizeof...(Args);
}

#define DEFINE_STATIC_DISPATCH(name)                                           \
    struct name##Impl {                                                        \
        static auto constexpr kFuncPtr = name;                                 \
        static auto constexpr kNoArgs = CountFunctionArgs(name);               \
        template <typename... Args> static auto Invoke(Args &&...args) {       \
            return name(std::forward<Args>(args)...);                          \
        }                                                                      \
    }

/**
 * This bunch of macros are aimed to define PyTorch-functions and bind it to
 * actual implementation.
 */

#define DEFINE_CONTINOUS_TORCH_FUNC_BODY(name, ...)                            \
    return ContinousCudaFunction<name##Impl>::apply(                           \
        inputs, bounds, levels __VA_OPT__(, __VA_ARGS__))

#define DEFINE_CONTINOUS_TORCH_FUNC0(name)                                     \
    DEFINE_STATIC_DISPATCH(name);                                              \
    DECLARE_CONTINOUS_TORCH_FUNC(name) {                                       \
        DEFINE_CONTINOUS_TORCH_FUNC_BODY(name);                                \
    }

#define DEFINE_CONTINOUS_TORCH_FUNC1(name, type1)                              \
    DEFINE_STATIC_DISPATCH(name);                                              \
    DECLARE_CONTINOUS_TORCH_FUNC(name, type1 arg1) {                           \
        DEFINE_CONTINOUS_TORCH_FUNC_BODY(name, arg1);                          \
    }

#define DEFINE_CONTINOUS_TORCH_FUNC2(name, type1, type2)                       \
    DEFINE_STATIC_DISPATCH(name);                                              \
    DECLARE_CONTINOUS_TORCH_FUNC(name, type1 arg1, type2 arg2) {               \
        DEFINE_CONTINOUS_TORCH_FUNC_BODY(name, arg1, arg2);                    \
    }

DEFINE_CONTINOUS_TORCH_FUNC1(Celu, double);
DEFINE_CONTINOUS_TORCH_FUNC1(Elu, double);
DEFINE_CONTINOUS_TORCH_FUNC0(Gelu);
DEFINE_CONTINOUS_TORCH_FUNC0(Hardswish);
DEFINE_CONTINOUS_TORCH_FUNC0(LogSigmoid);
DEFINE_CONTINOUS_TORCH_FUNC0(Mish);
DEFINE_CONTINOUS_TORCH_FUNC0(Selu);
DEFINE_CONTINOUS_TORCH_FUNC0(Sigmoid);
DEFINE_CONTINOUS_TORCH_FUNC0(Silu);
DEFINE_CONTINOUS_TORCH_FUNC2(Softplus, double, double);
DEFINE_CONTINOUS_TORCH_FUNC0(Softsign);
DEFINE_CONTINOUS_TORCH_FUNC0(Tanh);
DEFINE_CONTINOUS_TORCH_FUNC0(Tanhshrink);

TORCH_LIBRARY_IMPL(fewbit, AutogradCUDA, m) {
    // Stepwise functions.
    m.impl("hardshrink", fewbit::HardshrinkCuda);
    m.impl("hardsigmoid", fewbit::HardsigmoidCuda);
    m.impl("hardtanh", fewbit::HardtanhCuda);
    m.impl("leaky_relu", fewbit::LeakyReluCuda);
    m.impl("relu", fewbit::ReluCuda);
    m.impl("relu6", fewbit::Relu6Cuda);
    m.impl("softshrink", fewbit::SoftshrinkCuda);
    m.impl("threshold", fewbit::ThresholdCuda);

    // Continous functions.
    m.impl("celu", fewbit::CeluCuda);
    m.impl("elu", fewbit::EluCuda);
    m.impl("gelu", fewbit::GeluCuda);
    m.impl("hardswish", fewbit::HardswishCuda);
    m.impl("logsigmoid", fewbit::LogSigmoidCuda);
    m.impl("mish", fewbit::MishCuda);
    m.impl("selu", fewbit::SeluCuda);
    m.impl("sigmoid", fewbit::SigmoidCuda);
    m.impl("silu", fewbit::SiluCuda);
    m.impl("softplus", fewbit::SoftplusCuda);
    m.impl("softsign", fewbit::SoftsignCuda);
    m.impl("tanh", fewbit::TanhCuda);
    m.impl("tanhshrink", fewbit::TanhshrinkCuda);
}

} // namespace fewbit
