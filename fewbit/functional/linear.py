#   encoding: utf-8
#   filename: linear.py

import torch as T
import torch.autograd
import torch.nn.functional as F

from typing import Literal, Optional

from ..fft import dct

__all__ = ('linear_crs', 'linear_grp', 'linear_randomized')

MatMulType = Literal['dct', 'dft', 'gaussian', 'rademacher']


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


linear_crs = LinearCRSFunc.apply


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
    def forward(ctx, input: T.Tensor, weight: T.Tensor,
                bias: Optional[T.Tensor], proj_dim_ratio: Optional[float],
                proj_dim: Optional[int], proj_dim_max: Optional[int],
                proj_dim_min: Optional[int], matmul: MatMulType,
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

        if matmul == 'dct':
            proj_probas = T.ones(input_view.shape[0], device=device)
            proj_probas /= input_view.shape[0]
            proj = T.multinomial(input=proj_probas,
                                 num_samples=proj_features,
                                 replacement=True,
                                 generator=generator)
            proj_factor = proj_features * input_view.shape[0]
            proj_input = dct(input_view, dim=0, norm='ortho')
            proj_input = proj_factor * proj_input[proj, ...]
        elif matmul == 'dft':
            proj_probas = T.ones(input_view.shape[0], device=device)
            proj_probas /= input_view.shape[0]
            proj = T.multinomial(input=proj_probas,
                                 num_samples=proj_features,
                                 replacement=True,
                                 generator=generator)
            proj_factor = proj_features * input_view.shape[0]
            proj_input = T.fft.fft(input_view, dim=0, norm='ortho')
            proj_input = proj_factor * proj_input[proj, ...]
        elif matmul == 'gaussian':
            proj = T.randn((proj_features, input_view.shape[0]),
                           generator=generator,
                           device=device)
            proj_input = (proj @ input_view) / proj_features
        elif matmul == 'rademacher':
            proj_shape = (proj_features, input_view.shape[0])
            proj = T.randint(high=2,
                             size=proj_shape,
                             generator=generator,
                             device=generator.device,
                             dtype=input.dtype)
            proj = proj - 0.5
            proj_input = (proj @ input_view) * (4 / proj_features)
        else:
            raise ValueError(f'Unexpected matmul type: {matmul}.')

        ctx.save_for_backward(proj_input, weight, bias)
        ctx.proj_features = proj_features
        ctx.matmul = matmul
        ctx.generator_state = generator_state
        return F.linear(input, weight, bias)

    @staticmethod
    def backward(ctx, grad_output):
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

            # Depending on type of approximate mamul used we select specific
            # branch to rematerialize random projection.
            if ctx.matmul == 'dct':
                proj_probas = T.ones(grad_output_view.shape[0],
                                     device=grad_output.device)
                proj_probas /= grad_output_view.shape[0]
                proj = T.multinomial(input=proj_probas,
                                     num_samples=ctx.proj_features,
                                     replacement=True,
                                     generator=generator)
                grad_output_proj = dct(grad_output_view, dim=0,
                                       norm='ortho')[proj, :]
            elif ctx.matmul == 'dft':
                proj_probas = T.ones(grad_output_view.shape[0],
                                     device=grad_output.device)
                proj_probas /= grad_output_view.shape[0]
                proj = T.multinomial(input=proj_probas,
                                     num_samples=ctx.proj_features,
                                     replacement=True,
                                     generator=generator)
                grad_output_proj = T.fft.ifft(grad_output_view,
                                              dim=0,
                                              norm='ortho')[proj, :]
            elif ctx.matmul == 'gaussian':
                proj = T.randn((ctx.proj_features, grad_output_view.shape[0]),
                               generator=generator,
                               device=grad_output.device)
                grad_output_proj = proj @ grad_output_view
            elif ctx.matmul == 'rademacher':
                proj_shape = (ctx.proj_features, grad_output_view.shape[0])
                proj = T.randint(high=2,
                                 size=proj_shape,
                                 generator=generator,
                                 device=generator.device,
                                 dtype=input_proj.dtype)
                proj = proj - 0.5
                grad_output_proj = proj @ grad_output_view
            else:
                raise RuntimeError('Unexpected code path.')

            # Finally, estimate gradients of weights.
            if grad_output_proj.is_complex():
                grad_output_proj = grad_output_proj.real
            grad_weight = grad_output_proj.T @ input_proj

        grad_bias = None
        if ctx.needs_input_grad[2]:
            grad_bias = T.einsum('...i->i', grad_output)

        return (grad_input, grad_weight, grad_bias) + (None, ) * 6


linear_grp = LinearGRPFunc.apply
linear_randomized = linear_grp  # Use prettier naming.
