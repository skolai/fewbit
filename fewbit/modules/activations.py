#   encoding: utf-8
#   filename: module.py

import re

import torch as T

from inspect import Parameter, Signature, signature
from sys import modules
from typing import Optional, Tuple

from .. import functional
from ..functional.activations import stepwise

# Stepwise activation functions.
STEPWISE = ('Hardshrink', 'Hardsigmoid', 'Hardtanh', 'LeakyReLU', 'ReLU',
            'ReLU6', 'Softshrink', 'Stepwise', 'Threshold')

# Continous activation functions.
CONTINOUS = ('CELU', 'ELU', 'GELU', 'Hardswish', 'LogSigmoid', 'Mish', 'SELU',
             'Sigmoid', 'SiLU', 'Softplus', 'Softsign', 'Tanh', 'Tanhshrink')

__all__ = STEPWISE + CONTINOUS

re_remove_image = re.compile(r'^\s*\.\. image.*$', re.M)
re_subst_fewbit = re.compile(r'^(\s*>>> .*?)nn\.(.*)$', re.M)
re_args = re.compile(r'^(\s*)Args:\s*?$', re.M)
re_arg = re.compile(r'^(\s*)(\w+)(:|\s*?\().*?$', re.M)
re_shape = re.compile(r'^(\s*)Shape:\s*?$', re.M)
re_blank_line = re.compile(r'\s*$', re.M)

BITS_ARG = 'bits: number of bits in gradient approximation: Default: 3\n'
BITS_ARGS = '\n    Args:\n        ' + BITS_ARG
SEE_ALSO = """
    See Also:
        :class:`torch.nn.{name}` -- Original PyTorch implementation.
"""


def append_reference(doc: str, name: str) -> str:
    return doc + SEE_ALSO.format(name=name)


def insert_args(doc: str) -> str:
    # Try to find `Shape` section. If we failed then append arguments to end.
    if (res := re_shape.search(doc)) is None:
        doc += BITS_ARGS
        return doc
    beg = res.start(0)
    doc = doc[:beg] + BITS_ARGS + doc[beg:]
    return doc


def update_args(doc: str, beg: int, prefix: str = '    ') -> str:
    prefix += '    '
    # Ha-ha. Assume that there is not activation functions with more than ten
    # arguments.
    for i in range(10):
        if (res := re_blank_line.match(doc, beg)) is not None:
            beg, end = res.span(0)
            break
        if (res := re_arg.search(doc, beg)) is None:
            return doc
        prefix = res.group(1)
        beg = res.end(0) + 1
        if res.group(2) in ('approximate', 'inplace'):
            beg, end = res.span(0)
            doc = doc[:beg] + doc[end + 1:]
    doc = doc[:beg] + (prefix + BITS_ARG) + doc[beg:]
    return doc


def edit_doc(doc: str, name: str) -> str:
    """Function edit_doc generates docstring for custom non-linear activation
    function implementation from a docstring of original implementation. The
    algorithm is

    1. remove links to assets (images);
    2. update usage examples;
    3. add `bits` keyword argument to argument list;
    4. append 'See Also' section.
    """
    doc = re_remove_image.sub('', doc)
    doc = re_subst_fewbit.sub(r'\1fewbit.\2', doc)
    try:
        # Try to find and update `Args` section. Otherwise, find `Shape`
        # section and insert `Args` above `Shape`.
        if (res := re_args.search(doc)):
            doc = update_args(doc, res.end(0) + 1, res.group(1))
        else:
            doc = insert_args(doc)
        doc = append_reference(doc, name)
    finally:
        return doc


class Stepwise(T.nn.Module):
    """Class Stepwise provides customization ability to define your own
    stepwise approximation for in-place activation functions.

    :param borders: Internal borders of intervals.
    :param levels: Values of constant pieces.
    :param parity: Whether stepwise function is odd or even under shift
                   transformation.
    :param shift: Shift of the origin.
    """

    def __init__(self,
                 borders: T.Tensor,
                 levels: T.Tensor,
                 parity: Optional[bool] = None,
                 shift: Optional[Tuple[float, float]] = None):
        if borders.ndim != 1 or levels.ndim != 1:
            raise ValueError('Exepected number of dimensions of `borders` '
                             'and `levels` is one.')

        if borders.numel() > levels.numel():
            borders = borders[1:-1]

        if borders.numel() + 1 != levels.numel():
            raise ValueError('Size of `borders` should be lesser than size '
                             'of `levels` by one.')

        if levels.numel() > 256:
            raise ValueError('Maximal number of step limited to 256.')

        super().__init__()
        self.register_buffer('borders', borders, True)
        self.register_buffer('levels', levels, True)
        self.parity = parity
        self.shift = shift

    def forward(self, xs: T.Tensor) -> T.Tensor:
        return stepwise(xs, self.borders, self.levels, self.parity, self.shift)


class BuiltInStepwiseFunction(T.nn.Module):
    """Class BuiltInStepwiseFunction does some metaclass staff in order to
    coalesce custom in-place implementations of activation functions with
    PyTorch built-in ones.POSITIONAL_ONLY

    :param sig: Signature of original initialization method.
    :param args: Actual positional arguments.
    :param args: Actual keyword arguments.
    """

    PARAM_BITS = Parameter('bits',
                           Parameter.KEYWORD_ONLY,
                           default=None,
                           annotation=Optional[int])

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        if cls.__name__ != 'LeakyReLU':
            impl_name = cls.__name__.lower()
        else:
            impl_name = 'leaky_relu'

        cls_ref = getattr(T.nn, cls.__name__)  # Reference PyTorch class.
        sig = signature(cls_ref.__init__)
        sig = sig.replace(parameters=[
            param for param in sig.parameters.values()
            if param.name not in ('approximate', 'inplace')
        ] + [BuiltInStepwiseFunction.PARAM_BITS])

        def __init__(self, *args, **kwargs):
            super(cls, self).__init__(sig, *args, **kwargs)

        cls.__init__ = __init__
        cls.__init__.__signature__ = sig
        cls.__doc__ = edit_doc(cls_ref.__doc__, cls_ref.__name__)
        cls._impl_name = impl_name
        cls._impl = staticmethod(getattr(functional, cls._impl_name))

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
    if not hasattr(modules[__name__], name):
        ty = type(name, (BuiltInStepwiseFunction, ), {})
        setattr(modules[__name__], name, ty)
del name, ty
