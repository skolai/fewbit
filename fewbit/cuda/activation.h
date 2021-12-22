/**
 * \file activation.h
 *
 * \brief In this header file we defines PyTorch-functions for computing popular
 * activation functions on CUDA. We use macro definitions to avoid unnecessary
 * repetitions.
 */

#pragma once

#include <torch/torch.h>

namespace fewbit {

#ifdef DECLARE_STEPWISE_TORCH_FUNC
#error "Macro for declaration of stepwise functions is already defined."
#endif
#define DECLARE_STEPWISE_TORCH_FUNC(name, ...)                                 \
    torch::Tensor name##Cuda(                                                  \
        torch::Tensor const &inputs __VA_OPT__(, __VA_ARGS__));

DECLARE_STEPWISE_TORCH_FUNC(Hardshrink, double lambda = 0.5);
DECLARE_STEPWISE_TORCH_FUNC(Hardsigmoid);
DECLARE_STEPWISE_TORCH_FUNC(Hardtanh, double min_val = -1.0,
                            double max_val = 1.0);
DECLARE_STEPWISE_TORCH_FUNC(LeakyRelu, double negative_slope = 0.01);
DECLARE_STEPWISE_TORCH_FUNC(Relu);
DECLARE_STEPWISE_TORCH_FUNC(Relu6);
DECLARE_STEPWISE_TORCH_FUNC(Softshrink, double lambda = 0.5);
DECLARE_STEPWISE_TORCH_FUNC(Threshold, double threshold, double value);

#ifdef DECLARE_CONTINOUS_TORCH_FUNC
#error "Macro for declaration of countinous functions is already defined."
#endif
#define DECLARE_CONTINOUS_TORCH_FUNC(name, ...)                                \
    torch::Tensor name##Cuda(                                                  \
        torch::Tensor const &inputs, torch::Tensor const &bounds,              \
        torch::Tensor const &levels __VA_OPT__(, __VA_ARGS__))

DECLARE_CONTINOUS_TORCH_FUNC(Celu, double alpha = 1.0);
DECLARE_CONTINOUS_TORCH_FUNC(Elu, double alpha = 1.0);
DECLARE_CONTINOUS_TORCH_FUNC(Gelu);
DECLARE_CONTINOUS_TORCH_FUNC(Hardswish);
DECLARE_CONTINOUS_TORCH_FUNC(LogSigmoid);
DECLARE_CONTINOUS_TORCH_FUNC(Mish);
DECLARE_CONTINOUS_TORCH_FUNC(Selu);
DECLARE_CONTINOUS_TORCH_FUNC(Sigmoid);
DECLARE_CONTINOUS_TORCH_FUNC(Silu);
DECLARE_CONTINOUS_TORCH_FUNC(Softplus, double beta = 1, double threshold = 20);
DECLARE_CONTINOUS_TORCH_FUNC(Softsign);
DECLARE_CONTINOUS_TORCH_FUNC(Tanh);
DECLARE_CONTINOUS_TORCH_FUNC(Tanhshrink);

} // namespace fewbit
