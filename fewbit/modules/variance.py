"""Module variance contains modules for variance estimation of gradients of
linear layer.
"""

import torch as T


def override(f):
    return f


def estimate_variance_sgd(input: T.Tensor, output: T.Tensor) -> T.Tensor:
    bs = input.shape[0]
    fst = bs / (bs - 1)
    snd = 1 / (bs - 1)

    xs = T.linalg.norm(input, dim=1) ** 2
    ys = T.linalg.norm(output, dim=1) ** 2
    xy = T.linalg.norm(input.T @ output) ** 2
    return fst * (xs @ ys) - snd * xy


def estimate_variance_rmm(input: T.Tensor, output: T.Tensor) -> T.Tensor:
    bs = input.shape[0]
    xs = T.linalg.norm(input) ** 2
    ys = T.linalg.norm(output) ** 2
    xy = T.linalg.norm(input.T @ output) ** 2
    return (xs * ys - xy) / bs


def estimate_correlation(input: T.Tensor, output: T.Tensor) -> T.Tensor:
    pass


class GradientStorage:

    def __init__(self):
        self.input = None
        self.grad_output = None

    def forward(self, input):
        self.input = input.detach().clone()
        self.grad_output = None

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

    @override
    def postprocess(self):
        if self.input is None or self.grad_output is None:
            return

        input = self.input.reshape(-1, self.input.shape[-1])
        output = self.grad_output.reshape(-1, self.grad_output.shape[-1])

        var = (estimate_variance_sgd(input, output),
               estimate_variance_rmm(input, output))

        if callable(self.callback):
            self.callback(var, self.step)

        self.step += 1
        self.variance = var

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
        self.state.forward(input)
        output = self.model(input, *args, **kwargs)
        if isinstance(output, tuple):
            return catch_gradients(output[0], self.state), *output[1:]
        else:
            return catch_gradients(output, self.state)
