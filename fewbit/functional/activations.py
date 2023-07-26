#   encoding: utf-8
#   filename: functional.py

import numpy as np
import torch as T

from functools import partial, wraps
from inspect import Signature, Parameter, signature
from pathlib import Path
from sys import modules
from typing import Optional, Tuple, Union

# Stepwise activation functions.
STEPWISE = ('hardshrink', 'hardsigmoid', 'hardtanh', 'leaky_relu', 'relu',
            'relu6', 'softshrink', 'stepwise', 'threshold')

# Continous activation functions.
CONTINOUS = ('celu', 'elu', 'gelu', 'hardswish', 'logsigmoid', 'mish', 'selu',
             'sigmoid', 'silu', 'softplus', 'softsign', 'tanh', 'tanhshrink')

__all__ = STEPWISE + CONTINOUS + ('store', )


class StepwiseStore:
    """Class StepwiseStore is a singleton object to store and cache stepwise
    approximation for gradients of activation functions.
    """

    STORE = {}
    CACHE = {}

    def __len__(self) -> int:
        return len(self.STORE)

    def __repr__(self) -> str:
        stored = len(self.STORE)
        cached = len(self.CACHE)
        return f'{self.__class__.__name__}(stored={stored}, cached={cached})'

    def add(self, name: str, bits: int, value: Tuple[T.Tensor, T.Tensor]):
        borders, values = value
        key = (name, bits, T.device(borders.device), borders.dtype)
        self.STORE[key[:2]] = (borders, values.to(borders))
        self.CACHE[key] = self.STORE[key[:2]]

    def get(self,
            name: str,
            bits: int,
            device: Union[None, str, T.device] = None,
            dtype: Optional[T.dtype] = None):
        device = T.device(device or 'cpu')
        dtype = dtype or T.float32
        key = (name, bits, device, dtype)
        if (leaf := self.CACHE.get(key, None)):
            return leaf
        if (leaf := self.STORE.get(key[:2], None)):
            leaf_cached = tuple(el.to(device, dtype) for el in leaf)
            self.CACHE[key] = leaf_cached
            return leaf_cached
        raise KeyError(f'There is not {bits}-bit quantized gradients for '
                       f'activation function {name}.')

    def items(self, cached=False):
        if cached:
            yield from self.CACHE.items()
        else:
            yield from self.STORE.items()

    def load(self, path: Path) -> 'StepwiseStore':
        """Method load adds quantizations from npz-formatted file. We assume
        that all arrays in the file are named according to pattern
        `{func}{bits}-borders` or `{func}{bits}-levels`.

        :param path: Path to file to load.
        """
        with np.load(path) as npz:
            for key in {key.split('-', 1)[0] for key in npz.keys()}:
                name, bits = key[:-2], int(key[-2:])
                value = (T.tensor(npz[f'{key}-borders']),
                         T.tensor(npz[f'{key}-levels']))
                self.add(name, bits, value)


# Instantiate "singleton" object to manage quantization right here.
store = StepwiseStore()
store.load(Path(__file__).parent / '../data/builtin.npz')


class FallbackFunc(T.autograd.Function):
    """Class FallbackFunc is a fallback implementation of GELU activation
    function in pure Python.

    Attributes
    ----------
        _impl: actual implementation of forward pass.
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        if not cls.__name__.startswith('LeakyReLU'):
            impl_name = cls.__name__[:-12].lower()
        else:
            impl_name = 'leaky_relu'

        cls._impl_name = impl_name
        cls._impl = getattr(T.nn.functional, cls._impl_name)

    @staticmethod
    def forward(ctx, impl, input: T.Tensor, borders: T.Tensor,
                levels: T.Tensor, *args, **kwargs):
        if borders.numel() + 1 != levels.numel():
            raise ValueError('Size of `borders` should be lesser than size '
                             'of `levels` by one.')
        state = T.searchsorted(borders, input.float()).type(T.uint8)
        ctx.save_for_backward(state, levels)
        ctx.nargs = 4 + len(args) + len(kwargs)
        # We assume that this PyTorch function is not directly callable and its
        # derived types have class attribute `_impl` which has a reference to
        # an actual implementation of forward pass. See PEP-3135 for details.
        #
        # [1]: https://peps.python.org/pep-3135/
        return __class__._impl(input, *args, **kwargs)

    @staticmethod
    def backward(ctx, grad_output):
        state, levels = ctx.saved_tensors
        grad_input = (levels[state.type(T.int64)] * grad_output, )
        return (None, ) + grad_input + (None, ) * (ctx.nargs - 2)


class StepwiseFallbackFunc(T.autograd.Function):
    """Class StepwiseFallbackFunc is a fallback implementation of stepwise
    activation function in pure Python.
    """

    @staticmethod
    def stepwise(input: T.Tensor, borders: T.Tensor, levels: T.Tensor):
        raise NotImplementedError

    _impl = stepwise
    _impl_name = 'stepwise'


def dispatch(name, wrapper):
    wrapped = getattr(T.nn.functional, name)
    wrapper = wrapper or getattr(T.ops.fewbit, name)

    # Construct function signature for wrapper.
    try:
        sig = signature(wrapped)
    except ValueError:
        sig = Signature([
            Parameter('input', Parameter.POSITIONAL_ONLY, annotation=T.Tensor)
        ])
    params = [
        param for param in sig.parameters.values()
        if param.name not in ('approximate', 'inplace')
    ]
    params.extend([
        Parameter('bits',
                  Parameter.KEYWORD_ONLY,
                  default=None,
                  annotation=Optional[int]),
        Parameter('borders',
                  Parameter.KEYWORD_ONLY,
                  default=None,
                  annotation=Optional[T.Tensor]),
        Parameter('values',
                  Parameter.KEYWORD_ONLY,
                  default=None,
                  annotation=Optional[T.Tensor]),
    ])
    sig = sig.replace(parameters=params)

    @wraps(wrapped, ('__annotations__', '__doc__', '__name__'))
    def forward_call(*args, **kwargs):
        # Prepare positional and keyword arguments for forwarding to calls.
        bound_args = []
        bound_kwargs = {}
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        for param_name, param in sig.parameters.items():
            if param.kind in (Parameter.POSITIONAL_ONLY,
                              Parameter.POSITIONAL_OR_KEYWORD):
                bound_args.append(bound.arguments[param_name])
            elif param.kind == Parameter.KEYWORD_ONLY:
                bound_kwargs[param_name] = bound.arguments[param_name]
            else:
                raise RuntimeError(f'Unexpected execution path: {param.kind}.')

        # The first argument is always input tensor.
        input = bound_args[0]

        # Decide what quantisation to use.
        use_builtin = bound_kwargs['bits'] is not None
        use_custom = all([bound_kwargs['borders'] is not None,
                          bound_kwargs['values'] is not None])  # yapf: disable

        if use_builtin and use_custom:
            raise ValueError('Either `bits` or `borders` and `values` should '
                             'be scpecifed not both.')

        bits_default = 3
        bits = bound_kwargs.pop('bits', None) or bits_default
        borders = bound_kwargs.pop('borders')
        values = bound_kwargs.pop('values')

        if use_builtin or not use_custom:
            borders, values = store.get(name, bits, input.device, input.dtype)

        return wrapper(input, borders[1:-1].to(input), values.to(input),
                       *bound_args[1:], **bound_kwargs)

    forward_call.__qualname__ = name
    forward_call.__signature__ = sig

    return forward_call


def load_func(name: str):
    func_fallback = getattr(modules[__name__], f'{name}_fallback')

    try:
        func = getattr(T.ops.fewbit, name)
    except RuntimeError:
        func = None

    def dispatch_impl(input: T.Tensor, *args, **kwargs):
        """Function func_impl dispatches a function invokation to it CUDA
        backend implementation if CUDA devices are availiable and the function
        implementation for CUDA backend exists.
        """
        if input.device.type == 'cuda':
            return func(input, *args, **kwargs)
        else:
            return func_fallback(input, *args, **kwargs)

    func_impl = dispatch_impl
    if func is None:
        # If there is nothing to dispatch, do not dispatch!
        func_impl = func_fallback

    if name in STEPWISE:
        return func_impl
    elif name in CONTINOUS:
        return dispatch(name, func_impl)
    else:
        raise RuntimeError(f'Unknown activation function: {name}.')


# Define fallbacks for activation functions.
for name in STEPWISE + CONTINOUS:
    ty_name = name.capitalize() + 'FallbackFunc'
    if (ty := getattr(modules[__name__], ty_name, None)) is None:
        #ty = type(ty_name, (FallbackFunc, ),
        #          {'_impl': getattr(T.nn.functional, name, None)})
        ty = FallbackFunc
        ty._impl = getattr(T.nn.functional, name, None)
    setattr(modules[__name__], ty.__name__, ty)
    setattr(modules[__name__], f'{name}_fallback', partial(ty.apply, ty._impl))

# Import all functions enumerated above manually at runtime.
for name in STEPWISE + CONTINOUS:
    setattr(modules[__name__], name, load_func(name))

del name, ty, ty_name  # Remove local variables.
