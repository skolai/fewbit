#include <torch/script.h>

#include <fewbit/fewbit.h>

TORCH_LIBRARY(fewbit, m) {
    m.def("quantize", fewbit::Quantize);
    m.def("quantize_backward", fewbit::QuantizeBackward);
}
