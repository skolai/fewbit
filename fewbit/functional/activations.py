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
        key = (name, bits, T.device(device or 'cpu'), dtype or T.float32)
        if (leaf := self.CACHE.get(key, None)):
            return leaf
        if (leaf := self.STORE.get(key[:2], None)):
            self.CACHE[key] = leaf
            return leaf
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


def get_operator(name: str):
    try:
        func = getattr(T.ops.fewbit, name, None)
    except RuntimeError:
        # Use stub function to make an error verbose while loading activation
        # functions eagerly but silent.
        func = partial(stub, name)
    return func


def stub(name, *args, **kwargs):
    raise RuntimeError(f'Failed to look up function {name}.')


# Import all functions enumerated above manually at runtime.
for name in STEPWISE:
    setattr(modules[__name__], name, get_operator(name))
for name in CONTINOUS:
    setattr(modules[__name__], name, dispatch(name, get_operator(name)))
del name
