"""Module variance contains utility functions for cathing gradients.
"""

import torch as T


__all__ = ('GradientStorage', 'catch_gradients')


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
