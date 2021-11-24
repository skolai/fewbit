#!/usr/bin/env python3

import torch as T
import fewbit

bounds = T.tensor([
    -2.39798704e+00, -7.11248159e-01, -3.26290283e-01, -1.55338428e-04,
    3.26182064e-01, 7.10855860e-01, 2.39811567e+00
])

levels = T.tensor([
    -0.00260009, -0.08883533, 0.1251944, 0.37204148, 0.6277958, 0.87466175,
    1.08880716, 1.00259936
])

xs = T.tensor([-1, 0, 1, float('nan')], dtype=T.float32)
ys, codes = fewbit.quantize(xs, bounds)
print(ys)
print(codes)
print()

outputs = T.ones(4)
inputs = fewbit.quantize_backward(outputs, codes, levels)
print(outputs)
print(inputs)
