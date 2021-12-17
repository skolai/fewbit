#   encoding: utf-8
#   filename: module.py

import torch as T

from inspect import Parameter, Signature, signature
from sys import modules

from . import functional

# Stepwise activation functions.
STEPWISE = ('Hardshrink', 'Hardsigmoid', 'Hardtanh', 'LeakyReLU', 'ReLU',
            'ReLU6', 'Softshrink', 'Threshold')

# Continous activation functions.
CONTINOUS = ('CELU', 'ELU', 'GELU', 'Hardswish', 'LogSigmoid', 'Mish', 'SELU',
             'Sigmoid', 'SiLU', 'Softplus', 'Softsign', 'Tanh', 'Tanhshrink')

__all__ = STEPWISE + CONTINOUS


class BuiltInStepwiseFunction(T.nn.Module):
    """Class BuiltInStepwiseFunction does some metaclass staff in order to
    coalesce custom in-place implementations of activation functions with
    PyTorch built-in ones.POSITIONAL_ONLY

    :param sig: Signature of original initialization method.
    :param args: Actual positional arguments.
    :param args: Actual keyword arguments.
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        if cls.__name__ != 'LeakyReLU':
            impl_name = cls.__name__.lower()
        else:
            impl_name = 'leaky_relu'

        cls_ref = getattr(T.nn, cls.__name__)  # Reference PyTorch class.
        sig = signature(cls_ref.__init__)
        sig = sig.replace(parameters=[
            p for p in sig.parameters.values() if p.name != 'inplace'
        ])

        def __init__(self, *args, **kwargs):
            super(cls, self).__init__(sig, *args, **kwargs)

        cls.__init__ = __init__
        cls.__init__.__signature__ = sig
        cls._impl_name = impl_name
        cls._impl = getattr(functional, cls._impl_name)

    def __init__(self, sig: Signature, *args, **kwargs):
        super().__init__()

        # Set all arguments as instance attributes.
        bound = sig.bind(self, *args, **kwargs)
        bound.apply_defaults()
        bound.arguments.pop('self')
        for name, value in bound.arguments.items():
            setattr(self, name, value)  # Read-only attributes.

        # Prepare positional and keyword arguments for forwarding to calls.
        self.args = []
        self.kwargs = {}
        self.reprs = []
        for name, param in sig.parameters.items():
            if name == 'self':
                continue
            elif param.kind in (Parameter.POSITIONAL_ONLY,
                                Parameter.POSITIONAL_OR_KEYWORD):
                self.args.append(bound.arguments[name])
            elif param.kind == Parameter.KEYWORD_ONLY:
                self.kwargs[name] = bound.arguments[name]
            else:
                raise RuntimeError(f'Unexpected execution path: {param.kind}.')
            self.reprs.append(f'{name}={bound.arguments[name]}')

    def __repr__(self) -> str:
        args = ', '.join(self.reprs)
        return f'{self.__class__.__name__}({args})'

    def forward(self, xs: T.Tensor) -> T.Tensor:
        return self._impl(xs, *self.args, **self.kwargs)


# Produce PyTorch modules for in-place alternatives for built-in PyTorch
# activation function enumerated above manually at runtime.
for name in __all__:
    ty = type(name, (BuiltInStepwiseFunction, ), {})
    setattr(modules[__name__], name, ty)
del name, ty
