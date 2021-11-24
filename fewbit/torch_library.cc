#include <torch/script.h>

#include <fewbit/fewbit.h>

TORCH_LIBRARY(fewbit, m) {
    m.def("quantize", fewbit::Quantize);
    m.def("quantize_backward", fewbit::QuantizeBackward);

    m.def("gelu(Tensor self, Tensor bounds, Tensor levels) -> Tensor");
    m.impl("gelu", torch::DispatchKey::CPU, fewbit::GeluCpu);
}
