#include "gelu.h"

namespace fewbit {

namespace {

__global__ void gelu_kernel(int numel, int nobits, float const *bounds,
                            float const *inputs, float *output, float *buffer) {
}

__global__ void gelu_backward_kernel(int numel, int nobits, float const *bounds,
                                     float const *inputs, float *output,
                                     float *buffer) {
}

} // namespace

std::tuple<torch::Tensor, torch::Tensor>
gelu_cuda(torch::Tensor const &inputs, torch::Tensor const &bounds) {
    return {};
}

torch::Tensor gelu_backward_cuda(torch::Tensor const &grads,
                                 torch::Tensor const &buffer,
                                 torch::Tensor const &levels) {
    return {};
}

} // namespace fewbit
