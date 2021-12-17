#   encoding: utf-8
#   filename: functional.py

import torch as T

from functools import partial
from sys import modules

# Stepwise activation functions.
STEPWISE = ('hardshrink', 'hardsigmoid', 'hardtanh', 'leaky_relu', 'relu',
            'relu6', 'softshrink', 'stepwise', 'threshold')

# Continous activation functions.
CONTINOUS = ('celu', 'elu', 'gelu', 'hardswish', 'logsigmoid', 'mish', 'selu',
             'sigmoid', 'silu', 'softplus', 'softsign', 'tanh', 'tanhshrink')

__all__ = STEPWISE + CONTINOUS


def stub(name, *args, **kwargs):
    raise RuntimeError(f'Failed to look up function {name}.')


# Import all functions enumerated above manually at runtime.
for name in __all__:
    try:
        func = getattr(T.ops.fewbit, name, None)
    except RuntimeError:
        # Use stub function to make an error verbose while loading activation
        # functions eagerly but silent.
        func = partial(stub, name)
    finally:
        setattr(modules[__name__], name, func)
del func, name
