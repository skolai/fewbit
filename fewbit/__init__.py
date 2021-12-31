#   encoding: utf-8
#   filename: __init__.py

import torch as T

from pathlib import Path
from warnings import warn

try:
    T.ops.load_library(Path(__file__).with_name('libfewbit.so'))
except Exception as e:
    warn(f'Failed to load ops library: {e}.', RuntimeWarning)
finally:
    del Path, T, warn

from . import functional  # noqa: F401

from .approx import StepWiseFunction, approximate  # noqa: F401

# Import piecewise activation functions.
from .modules import (  # noqa: F401
    Hardshrink, Hardsigmoid, Hardtanh, LeakyReLU, ReLU, ReLU6, Softshrink,
    Threshold)

# Import continous activation functions.
from .modules import (  # noqa: F401
    CELU, ELU, GELU, Hardswish, LogSigmoid, Mish, SELU, Sigmoid, SiLU,
    Softplus, Softsign, Tanh, Tanhshrink)
