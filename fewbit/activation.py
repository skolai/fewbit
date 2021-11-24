import torch as T

from functools import cache

from torch.autograd.function import Function, once_differentiable

__all__ = ('GELUOp', 'GELUPy', 'gelu_op', 'gelu_py')

BOUNDS = T.tensor([
    -2.39798704e+00, -7.11248159e-01, -3.26290283e-01, -1.55338428e-04,
    3.26182064e-01, 7.10855860e-01, 2.39811567e+00
], dtype=T.float64)

LEVELS = T.tensor([
    -0.00260009, -0.08883533, 0.1251944, 0.37204148, 0.6277958, 0.87466175,
    1.08880716, 1.00259936
], dtype=T.float64)


@cache
def get_bounds_like(tensor: T.Tensor):
    return BOUNDS.type(tensor.dtype).to(tensor.device)


@cache
def get_levels_like(tensor: T.Tensor):
    return LEVELS.type(tensor.dtype).to(tensor.device)


class GELUOp(Function):

    @staticmethod
    def forward(ctx, xs: T.Tensor):
        bounds = get_bounds_like(xs)
        ys, state = T.ops.fewbit.quantize(xs, bounds)
        ctx.save_for_backward(state)
        return ys

    @staticmethod
    @once_differentiable
    def backward(ctx, outgrads: T.Tensor):
        state, = ctx.saved_tensors
        levels = get_levels_like(outgrads)
        ingrads = T.ops.fewbit.quantize_backward(outgrads, state, levels)
        return ingrads


class GELUPy(Function):

    @staticmethod
    def forward(ctx, xs: T.Tensor):
        ys = T.nn.functional.gelu(xs)
        bounds = get_bounds_like(xs)
        codes = T.searchsorted(bounds, xs).type(T.uint8)
        ctx.save_for_backward(codes)
        return ys

    @staticmethod
    @once_differentiable
    def backward(ctx, gs: T.Tensor):
        codes, = ctx.saved_tensors
        levels = get_levels_like(gs)
        return levels[codes.type(T.int64)] * gs


gelu_op = GELUOp.apply
gelu_py = GELUPy.apply
