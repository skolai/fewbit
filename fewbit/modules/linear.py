#   encoding: utf-8
#   filename: linear.py

import torch as T
import torch.autograd
import torch.nn.functional as F

from typing import Optional

__all__ = ('LinearCRS', 'LinearGRP')


def clamp(val: int,
          minval: Optional[int] = None,
          maxval: Optional[int] = None) -> int:
    if minval:
        val = max(minval, val)
    if maxval:
        val = min(maxval, val)
    return val


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
                 proj_dim: Optional[int] = None) -> None:
        super().__init__(in_features, out_features, proj_dim, device, dtype)
        self.proj_dim: int = proj_dim or out_features // 2

    def forward(self, input: T.Tensor) -> T.Tensor:
        return linear_crs(input, self.weight, self.bias, self.proj_dim)

    def extra_repr(self) -> str:
        return f'{super().extra_repr()}, nopairs={self.nopairs}'


class LinearGRPFunc(T.autograd.Function):

    @staticmethod
    def calc_proj_dim(ndim: int, proj_dim_ratio: Optional[float],
                      proj_dim: Optional[int], proj_dim_max: Optional[int],
                      proj_dim_min: Optional[int]):
        if proj_dim:
            result = proj_dim
        elif proj_dim_ratio:
            result = int(proj_dim_ratio * ndim)
        else:
            result = ndim
        return clamp(result, proj_dim_min, proj_dim_max)

    @staticmethod
    def forward(ctx,
                input: T.Tensor,
                weight: T.Tensor,
                bias: Optional[T.Tensor],
                proj_dim_ratio: Optional[float] = None,
                proj_dim: Optional[int] = None,
                proj_dim_max: Optional[int] = None,
                proj_dim_min: Optional[int] = None,
                mode: Optional[str] = None,
                generator: Optional[T.Generator] = None) -> T.Tensor:
        # TODO: Remove dispatching and generalize forward passes. Provide
        # specializations in backward passes for different modes.
        if mode == 'batch':
            return LinearGRPFunc.forward_batch(ctx, input, weight, bias,
                                               proj_dim_ratio, proj_dim,
                                               proj_dim_max, proj_dim_min,
                                               generator)
        elif mode == 'features' or mode is None:
            return LinearGRPFunc.forward_features(ctx, input, weight, bias,
                                                  proj_dim)
        else:
            raise ValueError(f'Unexpected mode: {mode}.')

    @staticmethod
    def forward_batch(ctx,
                      input: T.Tensor,
                      weight: T.Tensor,
                      bias: Optional[T.Tensor],
                      proj_dim_ratio: Optional[float],
                      proj_dim: Optional[int],
                      proj_dim_max: Optional[int],
                      proj_dim_min: Optional[int],
                      generator: Optional[T.Generator]) -> T.Tensor:
        device = input.device

        if not generator:
            if device.type == 'cpu':
                generator = T.random.default_generator
            else:
                generator = T.Generator(device=device)
        generator_state = generator.get_state()

        # Calculate number of dimension of projection space.
        input_view = input.view(-1, input.shape[-1])
        proj_features = LinearGRPFunc.calc_proj_dim(input_view.shape[0],
                                                    proj_dim_ratio, proj_dim,
                                                    proj_dim_max, proj_dim_min)
        proj = T.randn((proj_features, input_view.shape[0]),
                       generator=generator,
                       device=device)
        proj_input = (proj @ input_view) / proj_features

        ctx.save_for_backward(proj_input, weight, bias)
        ctx.mode = 'batch'
        ctx.proj_features = proj_features
        ctx.generator_state = generator_state
        return F.linear(input, weight, bias)

    @staticmethod
    def forward_features(ctx, input: T.Tensor, weight: T.Tensor,
                         bias: Optional[T.Tensor],
                         proj_features: int) -> T.Tensor:
        in_features = weight.shape[1]
        proj = T.randn((proj_features, in_features))
        input_proj = (input @ proj.T) / proj_features
        ctx.save_for_backward(input_proj, weight, bias, proj)
        ctx.mode = 'features'
        return F.linear(input, weight, bias)

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.mode == 'batch':
            return LinearGRPFunc.backward_batch(ctx, grad_output)
        elif ctx.mode == 'features':
            return LinearGRPFunc.backward_features(ctx, grad_output)
        else:
            raise RuntimeError('Unexpected execution path.')

    @staticmethod
    def backward_batch(ctx, grad_output):
        input_proj, weight, bias = ctx.saved_tensors
        generator = T.Generator(grad_output.device)
        generator.set_state(ctx.generator_state)

        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output @ weight

        grad_weight = None
        if ctx.needs_input_grad[1]:
            # We are forced to reshape intead of view because output gradients
            # has incompatible size and strides.
            grad_output_view = grad_output.reshape(-1, grad_output.shape[-1])
            proj = T.randn((ctx.proj_features, grad_output_view.shape[0]),
                           generator=generator,
                           device=grad_output.device)
            grad_output_proj = proj @ grad_output_view
            grad_weight = grad_output_proj.T @ input_proj

        grad_bias = None
        if ctx.needs_input_grad[2]:
            grad_bias = T.einsum('...i->i', grad_output)

        return (grad_input, grad_weight, grad_bias) + (None, ) * 6

    @staticmethod
    def backward_features(ctx, grad_output):
        input_proj, weight, bias, proj = ctx.saved_tensors

        input_grad = None
        if ctx.needs_input_grad[0]:
            input_grad = grad_output @ weight

        weight_grad = None
        if ctx.needs_input_grad[1]:
            weight_grad = T.einsum('...i,...j->ij', input_proj, grad_output)
            weight_grad = weight_grad.T @ proj

        bias_grad = None
        if ctx.needs_input_grad[2]:
            bias_grad = T.einsum('...i->i', grad_output)

        return (input_grad, weight_grad, bias_grad) + (None, ) * 6


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
                 mode: Optional[str] = None,
                 generator: Optional[T.Generator] = None) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        self.generator = generator
        self.mode = mode or 'batch'
        self.proj_dim_ratio = proj_dim_ratio
        self.proj_dim = proj_dim
        self.proj_dim_max = proj_dim_max
        self.proj_dim_min = proj_dim_min

    def forward(self, input: T.Tensor) -> T.Tensor:
        return linear_grp(input, self.weight, self.bias, self.proj_dim_ratio,
                          self.proj_dim, self.proj_dim_max, self.proj_dim_min,
                          self.mode, self.generator)

    def extra_repr(self) -> str:
        return ', '.join([
            super().extra_repr(),
            f'mode={self.mode}',
            f'proj_dim={self.proj_dim}',
            f'proj_dim_ratio={self.proj_dim_ratio}',
            f'proj_dim_max={self.proj_dim_max}',
            f'proj_dim_min={self.proj_dim_min}',
        ])


linear_crs = LinearCRSFunc.apply
linear_grp = LinearGRPFunc.apply
