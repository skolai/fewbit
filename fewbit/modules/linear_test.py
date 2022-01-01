#   encoding: utf-8
#   filename: linear_test.py

from unittest import TestCase

import torch as T

from fewbit.modules.linear import LinearCRS, LinearGRP


def clone_module(module: T.nn.Module) -> T.nn.Module:
    cloned = T.nn.Linear(module.in_features, module.out_features,
                         module.bias is not None)
    for name, param in cloned.named_parameters():
        param.data = getattr(module, name).clone()
    return cloned


def estimate_grad(module: T.nn.Module, xs: T.Tensor, repeat: int = 1):
    weight_grad = T.zeros_like(module.weight)
    if (with_bias := module.bias is not None):
        bias_grad = T.zeros_like(module.bias)

    for _ in range(repeat):
        module.zero_grad()
        xs.grad = None
        ys = module(xs)
        ys.backward(T.ones_like(ys))
        weight_grad += module.weight.grad
        if with_bias:
            bias_grad += module.bias.grad

    weight_grad /= repeat
    if with_bias:
        bias_grad /= repeat

    if with_bias:
        return xs.grad, weight_grad, bias_grad
    else:
        return xs.grad, weight_grad, None


class LinearTestCaseMixin:

    def __init_subclass__(cls) -> None:
        cls.layer_name = cls.__name__.removeprefix('Test')
        cls.layer_ctor = globals().get(cls.layer_name, None)
        if cls.layer_ctor is None:
            raise RuntimeError(f'Layer {cls.layer_name} is not imported.')

    def setUp(self):
        self.rng = T.random.manual_seed(42)

    def test_forward(self):
        for bias in (False, True):
            with self.subTest(bias=bias):
                module = self.layer_ctor(8, 4, bias)
                module_cloned = clone_module(module)

                with T.no_grad():
                    xs = T.randn((128, 8))
                    ys = module(xs)
                    zs = module_cloned(xs)

                rel = T.linalg.norm(zs - ys) / T.linalg.norm(zs)
                self.assertAlmostEqual(0, rel.item(), delta=1e-6)

    def test_backward(self):
        for bias in (False, True):
            with self.subTest(bias=bias):
                module = self.layer_ctor(8, 4, bias)
                module_cloned = clone_module(module)

                xs = T.randn((128, 8), requires_grad=True)
                ys, ps, ss = estimate_grad(module, xs, 4096)
                zs, qs, ts = estimate_grad(module_cloned, xs)

                err_input = T.linalg.norm(ys - zs) / T.linalg.norm(zs)
                err_weight = T.linalg.norm(ps - qs) / T.linalg.norm(qs)
                self.assertAlmostEqual(0, err_input.item(), delta=1e-6)
                self.assertAlmostEqual(0, err_weight.item(), delta=5e-2)

                if bias:
                    err_bias = T.linalg.norm(ss - ts) / T.linalg.norm(ts)
                    self.assertAlmostEqual(0, err_bias.item(), delta=1e-6)


class TestLinearCRS(TestCase, LinearTestCaseMixin):
    pass


class TestLinearGRP(TestCase, LinearTestCaseMixin):
    pass
