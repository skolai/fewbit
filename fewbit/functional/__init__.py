#   encoding: utf-8
#   filename: __init__.py

# Import piecewise activation functions.
from .activations import (  # noqa: F401
    hardshrink, hardsigmoid, hardtanh, leaky_relu, relu, relu6, softshrink,
    threshold)

# Import continous activation functions.
from .activations import (  # noqa: F401
    celu, elu, gelu, hardswish, logsigmoid, mish, selu, sigmoid, silu,
    softplus, softsign, tanh, tanhshrink)

# Import linear layers with approximate matmul.
from .linear import linear_crs, linear_grp, linear_randomized  # noqa: F401

# Import utility routines for tracking gradients.
from .variance import GradientStorage, catch_gradients  # noqa: F401
