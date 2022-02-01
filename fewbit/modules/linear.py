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
    """Class LinearGRP defines PyTorch layer based torch.nn.Linear module. It
    approximate gradients (of weights) with Gaussian random projections (GRP)
    and stored subset of input columns for backward phase.
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
