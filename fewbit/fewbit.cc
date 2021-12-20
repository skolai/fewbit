#include "fewbit.h"

#include <torch/script.h>

TORCH_LIBRARY(fewbit, m) {
    m.def("quantize", fewbit::Quantize);
    m.def("quantize_backward", fewbit::QuantizeBackward);

    /* clang-format off */
    m.def("hardshrink (Tensor(a!) self, float lambda) -> Tensor(a!)");
    m.def("hardsigmoid(Tensor(a!) self) -> Tensor(a!)");
    m.def("hardtanh   (Tensor(a!) self, float min_val, float max_val) -> Tensor(a!)");
    m.def("leaky_relu (Tensor(a!) self, float negative_slope) -> Tensor(a!)");
    m.def("relu       (Tensor(a!) self) -> Tensor(a!)");
    m.def("relu6      (Tensor(a!) self) -> Tensor(a!)");
    m.def("softshrink (Tensor(a!) self, float lambda) -> Tensor(a!)");
    m.def("threshold  (Tensor(a!) self, float threshold, float value) -> Tensor(a!)");
    /* clang-format on */

    /* clang-format off */
    m.def("celu      (Tensor(a!) self, Tensor bounds, Tensor levels) -> Tensor(a!)");
    m.def("elu       (Tensor(a!) self, Tensor bounds, Tensor levels) -> Tensor(a!)");
    m.def("gelu      (Tensor(a!) self, Tensor bounds, Tensor levels) -> Tensor(a!)");
    m.def("hardswish (Tensor(a!) self, Tensor bounds, Tensor levels) -> Tensor(a!)");
    m.def("logsigmoid(Tensor(a!) self, Tensor bounds, Tensor levels) -> Tensor(a!)");
    m.def("mish      (Tensor(a!) self, Tensor bounds, Tensor levels) -> Tensor(a!)");
    m.def("selu      (Tensor(a!) self, Tensor bounds, Tensor levels) -> Tensor(a!)");
    m.def("sigmoid   (Tensor(a!) self, Tensor bounds, Tensor levels) -> Tensor(a!)");
    m.def("silu      (Tensor(a!) self, Tensor bounds, Tensor levels) -> Tensor(a!)");
    m.def("softplus  (Tensor(a!) self, Tensor bounds, Tensor levels) -> Tensor(a!)");
    m.def("softsign  (Tensor(a!) self, Tensor bounds, Tensor levels) -> Tensor(a!)");
    m.def("tanh      (Tensor(a!) self, Tensor bounds, Tensor levels) -> Tensor(a!)");
    m.def("tanhshrink(Tensor(a!) self, Tensor bounds, Tensor levels) -> Tensor(a!)");
    /* clang-format on */

    /* clang-format off */
    m.def("stepwise   (Tensor(a!) self, Tensor bounds, Tensor levels, bool? parity=None, int[2]? shift=None) -> Tensor(a!)");
    /* clang-format on */
}
