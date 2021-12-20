#pragma once

#include <torch/torch.h>

namespace fewbit {

torch::Tensor HardshrinkCuda(torch::Tensor const &inputs, double lambda = 0.5);

torch::Tensor HardsigmoidCuda(torch::Tensor const &inputs);

torch::Tensor HardtanhCuda(torch::Tensor const &inputs, double min_val = -1.0,
                           double max_val = 1.0);

torch::Tensor LeakyReluCuda(torch::Tensor const &inputs,
                            double negative_slope = 0.01);

torch::Tensor ReluCuda(torch::Tensor const &inputs);

torch::Tensor Relu6Cuda(torch::Tensor const &inputs);

torch::Tensor SoftshrinkCuda(torch::Tensor const &inputs, double lambda = 0.5);

torch::Tensor ThresholdCuda(torch::Tensor const &inputs, double threshold,
                            double value);

} // namespace fewbit
