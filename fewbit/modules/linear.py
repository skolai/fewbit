#   encoding: utf-8
#   filename: linear.py

import torch as T
import torch.autograd

from typing import Literal, Optional

from ..functional.linear import linear_crs, linear_grp

__all__ = ('LinearCRS', 'LinearGRP', 'RandomizedLinear')

MatMulType = Literal['dct', 'dft', 'gaussian', 'rademacher']


class LinearCRS(T.nn.Linear):
    """Class LinearCRS defines PyTorch layer based torch.nn.Linear module. It
    approximate gradients (of weights) with column-row sampling (CRS) and
    stored subset of input columns for backward phase.
    """

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 device=None,
                 dtype=None,
                 proj_dim: Optional[int] = None) -> None:
        super().__init__(in_features, out_features, proj_dim, device, dtype)
        self.proj_dim: int = proj_dim or out_features // 2

    def forward(self, input: T.Tensor) -> T.Tensor:
        return linear_crs(input, self.weight, self.bias, self.proj_dim)

    def extra_repr(self) -> str:
        return f'{super().extra_repr()}, nopairs={self.nopairs}'


class LinearGRP(T.nn.Linear):
    r"""Class ``RandomizedLinear`` defines PyTorch layer based
    :class:`torch.nn.Linear` module. It approximate gradients (of weights) with
    random projections and stored subset of input columns for backward phase as
    described in paper `Memory-Efficient Backpropagation through Large Linear
    Layers <https://arxiv.org/abs/2201.13195>`_ by Bershatsky et al. The layer
    applies a linear transformation to the incoming data: :math:`y = xA^T + b`
    with compression along batch dimension.

    Parameters
    ----------
    in_features : int
        Size of each input sample.
    out_features : int
        Size of each output sample.
    bias : bool, default=True
        If set to ``False``, the layer will not learn an additive bias.
    device : optional
        Device where to place layer parameters.
    dtype : optional
        Element type of bias vector and weight matrix.
    proj_dim_ratio : float, optional
        Compression ratio which is a ``proj_dim`` divided by batch size.
    proj_dim : int, optional
        Exact number of dimensions of projection space.
    proj_dim_min : int, optional
        Lower bound on number of dimensions of projection space.
    proj_dim_max : int, optional
        Upper bound on number of dimensions of projection space.
    matmul : {'dct', 'dft', 'gaussian', 'rademacher'}, default='gaussian'
        Type of random projection to use.
    generator : torch.random.Generator, optional
        Random number generator to use. If not specified that global one is
        used.


    Either ``proj_dim_ratio`` or ``proj_dim`` parameter should be specified.

    Shape:
        - Input: :math:`(*, H_{in})` where :math:`*` means any number of
          dimensions including none and :math:`H_{in} = \text{in\_features}`.
        - Output: :math:`(*, H_{out})` where all but the last dimension
          are the same shape as the input and
          :math:`H_{out} = \text{out\_features}`.

    Examples:

        >>> m = fewbit.RandomizedLinear(20, 30, proj_dim_ratio=0.5)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])

    Attributes
    ----------
    weight : torch.Tensor
        The learnable weights of the module of shape
        :math:`(\text{out\_features}, \text{in\_features})`. The values are
        initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
        :math:`k = \frac{1}{\text{in\_features}}`.
    bias : torch.Tensor
        The learnable bias of the module of shape
        :math:`(\text{out\_features})`. If :attr:`bias` is ``True``, the values
        are initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
        :math:`k = \frac{1}{\text{in\_features}}`.


    See Also:
        :class:`torch.nn.Linear` -- Original PyTorch implementation.
    """

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 device=None,
                 dtype=None,
                 proj_dim_ratio: Optional[float] = None,
                 proj_dim: Optional[int] = None,
                 proj_dim_min: Optional[int] = None,
                 proj_dim_max: Optional[int] = None,
                 matmul: MatMulType = 'gaussian',
                 generator: Optional[T.Generator] = None) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        self.generator = generator
        self.matmul = matmul
        self.proj_dim_ratio = proj_dim_ratio
        self.proj_dim = proj_dim
        self.proj_dim_max = proj_dim_max
        self.proj_dim_min = proj_dim_min

    def forward(self, input: T.Tensor) -> T.Tensor:
        return linear_grp(input, self.weight, self.bias, self.proj_dim_ratio,
                          self.proj_dim, self.proj_dim_max, self.proj_dim_min,
                          self.matmul, self.generator)

    def extra_repr(self) -> str:
        return ', '.join([
            super().extra_repr(),
            f'matmul={self.matmul}',
            f'proj_dim={self.proj_dim}',
            f'proj_dim_ratio={self.proj_dim_ratio}',
            f'proj_dim_max={self.proj_dim_max}',
            f'proj_dim_min={self.proj_dim_min}',
        ])


RandomizedLinear = LinearGRP  # Use prettier naming.
