#include <torch/script.h>

#include <fewbit/fewbit.h>

PYBIND11_MODULE(fewbit, m) {
    m.def("quantize", fewbit::Quantize);
    m.def("quantize_backward", fewbit::QuantizeBackward);
}
