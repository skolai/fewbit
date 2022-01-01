#   encoding: utf-8
#   filename: linear.py

import torch as T
import torch.autograd
import torch.nn.functional as F

from typing import Optional

__all__ = ('LinearCRS', 'LinearGRP')


class LinearCRSFunc(T.autograd.Function):

    @staticmethod
    def forward(ctx, input: T.Tensor, weight: T.Tensor,
                bias: Optional[T.Tensor], nopairs: int) -> T.Tensor:
        in_features = weight.shape[1]
        proba = 1 / in_features  # Uniform col-row distribution.
        factor = proba * nopairs

        randomness = T.randint(0, in_features, (nopairs, )).to(weight.device)
        bins = T.bincount(randomness, minlength=in_features)
        pairs, = T.nonzero(bins, as_tuple=True)
        counts = bins[pairs]
        scale = counts.type(T.float32) / factor
        input_proj = input[..., pairs] @ T.diag(scale)
        ctx.save_for_backward(input_proj, weight, bias, pairs)
        return F.linear(input, weight, bias)

    @staticmethod
    def backward(ctx, grad_output):
        input_proj, weight, bias, pairs = ctx.saved_tensors

        input_grad = None
        if ctx.needs_input_grad[0]:
            input_grad = grad_output @ weight

        weight_grad = None
        if ctx.needs_input_grad[1]:
            weight_grad = T.zeros_like(weight)
            weight_grad[..., pairs] = T.einsum('...i,...j->ij', grad_output,
                                               input_proj)

        bias_grad = None
        if ctx.needs_input_grad[2]:
            bias_grad = T.einsum('...i->i', grad_output)

        return (input_grad, weight_grad, bias_grad, None)


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
                 nopairs: Optional[int] = None) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        self.nopairs: int = nopairs or out_features // 2

    def forward(self, input: T.Tensor) -> T.Tensor:
        return linear_crs(input, self.weight, self.bias, self.nopairs)

    def extra_repr(self) -> str:
        return f'{super().extra_repr()} + nopairs={self.nopairs}'


class LinearGRPFunc(T.autograd.Function):

    @staticmethod
    def forward(ctx, input: T.Tensor, weight: T.Tensor,
                bias: Optional[T.Tensor], proj_features: int) -> T.Tensor:
        in_features = weight.shape[1]
        proj = T.randn((proj_features, in_features))
        input_proj = input @ proj.T
        ctx.save_for_backward(input_proj, weight, bias, proj)
        return F.linear(input, weight, bias)

    @staticmethod
    def backward(ctx, grad_output):
        input_proj, weight, bias, proj = ctx.saved_tensors

        input_grad = None
        if ctx.needs_input_grad[0]:
            input_grad = grad_output @ weight

        weight_grad = None
        if ctx.needs_input_grad[1]:
            weight_grad = np.einsum('...i,...j->ij', input_proj, grad_output)
            weight_grad = proj.T @ weight_grad

        bias_grad = None
        if ctx.needs_input_grad[2]:
            bias_grad = T.einsum('...i->i', grad_output)

        return (input_grad, weight_grad, bias_grad, None)


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
                 proj_features: Optional[int] = None) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        self.proj_features: int = proj_features or out_features // 2

    def forward(self, input: T.Tensor) -> T.Tensor:
        return linear_crs(input, self.weight, self.bias, self.proj_features)

    def extra_repr(self) -> str:
        return f'{super().extra_repr()} + proj_features={self.nopairs}'


linear_crs = LinearCRSFunc.apply
linear_grp = LinearGRPFunc.apply
