import torch as T
import torch.nn.functional as F

from unittest import TestCase, skip, skipUnless

from fewbit.functional import (hardshrink, hardsigmoid, hardtanh, leaky_relu,
                               relu, relu6, softshrink, threshold)

from fewbit.functional import store


@skipUnless(T.cuda.is_available(), 'CUDA support is required.')
class TestStepwiseFunctions(TestCase):

    def _test_parametric(self, lhs, rhs, *args, **kwargs):
        ps = self.xs.clone().requires_grad_()
        ys = lhs(ps, *args, **kwargs)
        ys.backward(self.gs)

        qs = self.xs.clone().requires_grad_()
        zs = rhs(qs.clone(), *args, **kwargs)
        zs.backward(self.gs)

        v_err = T.linalg.norm(zs - ys).item()
        g_err = T.linalg.norm(ps.grad - qs.grad).item()

        self.assertAlmostEqual(0, v_err, places=6)
        self.assertAlmostEqual(0, g_err, places=6)

    def setUp(self):
        self.xs = T.linspace(-5, 5, 101).to('cuda')
        self.gs = T.ones_like(self.xs)

    def test_hardshrink(self):
        self._test_parametric(F.hardshrink, hardshrink)
        self._test_parametric(F.hardshrink, hardshrink, lambd=1.0)

    def test_hardsigmoid(self):
        self._test_parametric(F.hardsigmoid, hardsigmoid)

    def test_hardtanh(self):
        self._test_parametric(F.hardtanh, hardtanh)
        self._test_parametric(F.hardtanh, hardtanh, min_val=-2.0, max_val=2.0)

    def test_leaky_relu(self):
        self._test_parametric(F.leaky_relu, leaky_relu)
        self._test_parametric(F.leaky_relu, leaky_relu, negative_slope=0.5)

    def test_relu(self):
        self._test_parametric(F.relu, relu)

    def test_relu6(self):
        self._test_parametric(F.relu6, relu6)

    def test_softshrink(self):
        self._test_parametric(F.softshrink, softshrink)
        self._test_parametric(F.softshrink, softshrink, lambd=1.0)

    @skip('Activation function is not implemented')
    def test_stepwise(self):
        pass

    def test_threshold(self):
        self._test_parametric(F.threshold, threshold, 1.0, 3.0)


class TestStepwiseStore(TestCase):

    def test_load(self):
        self.assertGreater(len(store), 0)
        self.assertIsNotNone(store.get('gelu', 3))
