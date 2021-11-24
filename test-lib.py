#!/usr/bin/env python3

import torch as T

T.ops.load_library('libfewbit.so')


@T.jit.script
def foo(inputs, outgrads, bounds, levels):
    outputs, codes = T.ops.fewbit.quantize(inputs, bounds)
    ingrads = T.ops.fewbit.quantize_backward(outgrads, codes, levels)
    return outputs, ingrads


print(foo.graph)
