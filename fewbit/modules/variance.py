"""Module variance contains modules for variance estimation of gradients of
linear layer.
"""

import numpy as np
import torch as T

from typing import Optional

from fewbit.modules.linear import LinearGRPFunc


def override(f):
    return f


def estimate_correlation(input: T.Tensor, output: T.Tensor) -> T.Tensor:
    xs = T.linalg.norm(input)
    ys = T.linalg.norm(output)
    xy = T.linalg.norm(input.T @ output)
    return (xy / (xs * ys))**2


def estimate_variance_sgd(input: T.Tensor, output: T.Tensor,
                          bs: Optional[int] = None) -> T.Tensor:
    if not bs:
        bs = input.shape[0]
    fst = bs / (bs - 1)
    snd = 1 / (bs - 1)
    xs = T.linalg.norm(input, dim=1)**2
    ys = T.linalg.norm(output, dim=1) ** 2
    xy = T.linalg.norm(input.T @ output) ** 2
    return fst * (xs @ ys) - snd * xy


def estimate_variance_rmm(input: T.Tensor, output: T.Tensor,
                          bs_proj: Optional[int] = None) -> T.Tensor:
    # NOTE We have to specify bs_proj since there is no way to inference it
    # from function arguments.
    if not bs_proj:
        bs_proj = input.shape[0]
    xs = T.linalg.norm(input) ** 2
    ys = T.linalg.norm(output) ** 2
    xy = T.linalg.norm(input.T @ output) ** 2
    return (xs * ys - xy) / bs_proj


class GradientStorage:
    """Class GradientStorage implements a minimal stateful object to accumulate
    input and output gradients for a module.
    """

    def __init__(self):
        self.input = None
        self.grad_output = None

    def forward(self, input):
        self.input = input.detach().clone()

    def backward(self, grad_output):
        self.grad_output = grad_output.detach().clone()
        self.postprocess()

    def postprocess(self):
        """Method postprocess should be overrided in children.
        """


class VarianceEstimatorImpl(GradientStorage):

    def __init__(self, callback=None):
        super().__init__()
        self.callback = callback
        self.step = 0
        self.variance = None
        self.bs = None
        self.bs_proj = None

    def set_batch_size(self, bs, bs_proj):
        self.bs = bs
        self.bs_proj = bs_proj

    @override
    def postprocess(self):
        if self.input is None or self.grad_output is None:
            return

        input = self.input.reshape(-1, self.input.shape[-1])
        output = self.grad_output.reshape(-1, self.grad_output.shape[-1])

        corr = estimate_correlation(input, output)
        var_sgd = estimate_variance_sgd(input, output, self.bs)
        var_rmm = estimate_variance_rmm(input, output, self.bs_proj)

        if callable(self.callback):
            self.callback(corr, var_sgd, var_rmm, self.step)

        self.step += 1
        self.variance = (corr, var_sgd, var_sgd)

    def __repr__(self) -> str:
        has_input = self.input is not None
        has_grad_output = self.grad_output is not None
        args = ', '.join([
            str(has_input),
            str(has_grad_output),
            f'{self.variance[0]:e}',
            f'{self.variance[1]:e}',
        ])
        return f'VarianceEstimatorImpl({args})'


class GradientCatcher(T.autograd.Function):

    @staticmethod
    def forward(ctx, input: T.Tensor, storage: GradientStorage) -> T.Tensor:
        ctx.storage = storage
        return input

    @staticmethod
    def backward(ctx, grad_output: T.Tensor):
        ctx.storage.backward(grad_output)
        return grad_output, None


catch_gradients = GradientCatcher.apply


class VarianceEstimator(T.nn.Module):

    def __init__(self, model: T.nn.Module, callback=None):
        super().__init__()
        self.model = model
        self.state = VarianceEstimatorImpl(callback)

    @property
    def variance(self):
        return self.state.variance

    def forward(self, input: T.Tensor, *args, **kwargs):
        bs = np.prod(input.shape[:-1])
        bs_proj = LinearGRPFunc.calc_proj_dim(bs, self.model.proj_dim_ratio,
                                              self.model.proj_dim,
                                              self.model.proj_dim_max,
                                              self.model.proj_dim_min)

        self.state.set_batch_size(bs, bs_proj)
        self.state.forward(input)

        output = self.model(input, *args, **kwargs)
        if isinstance(output, tuple):
            return catch_gradients(output[0], self.state), *output[1:]
        else:
            return catch_gradients(output, self.state)
