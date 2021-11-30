#include "fewbit.h"

#include <torch/script.h>

TORCH_LIBRARY(fewbit, m) {
    m.def("quantize", fewbit::Quantize);
    m.def("quantize_backward", fewbit::QuantizeBackward);

    m.def("gelu(Tensor self, Tensor bounds, Tensor levels) -> Tensor");
}
