import numpy as np
import torch as T
import torch.nn.functional as F

from unittest import TestCase, skip, skipUnless

from fewbit.approx import StepWiseFunction, estimate_error
from fewbit.functional import (hardshrink, hardsigmoid, hardtanh, leaky_relu,
                               relu, relu6, softshrink, threshold, celu, elu,
                               gelu, hardswish, logsigmoid, mish, selu,
                               sigmoid, silu, softplus, softsign, tanh,
                               tanhshrink, store)


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


@skipUnless(T.cuda.is_available(), 'CUDA support is required.')
class TestContinousFunctions(TestCase):

    def _test_parametric(self, lhs, rhs, nobits, *args, **kwargs):
        ps = self.xs.clone().requires_grad_()
        ys = lhs(ps, *args, **kwargs)

        qs = self.xs.clone().requires_grad_()
        zs = rhs(qs.clone(), *args, **kwargs)

        v_err = T.linalg.norm(zs - ys).item()
        self.assertAlmostEqual(0, v_err, delta=1e-6)

        # Estimate L2 approximation error.
        def deriv_lhs(xs: np.ndarray) -> np.ndarray:
            ys = T.tensor(xs).to(self.device).requires_grad_()
            zs = lhs(ys.clone(), *args, **kwargs)
            zs.backward(T.ones_like(ys))
            return ys.grad.cpu().numpy()

        if (entry_key := rhs.__name__) == 'log_sigmoid':
            entry_key = 'logsigmoid'
        entry = store.get(entry_key, nobits)
        deriv_rhs = StepWiseFunction(*[el.numpy() for el in entry])

        g_err, _ = estimate_error(deriv_lhs, deriv_rhs, 1e-2)
        self.assertAlmostEqual(0, g_err, delta=1e-1)

    def setUp(self):
        self.xs = T.linspace(-5, 5, 101).to('cuda')
        self.gs = T.ones_like(self.xs)
        self.device = self.xs.device

    def test_celu(self):
        self._test_parametric(F.celu, celu, 3)

    def test_elu(self):
        self._test_parametric(F.elu, elu, 3)

    def test_gelu(self):
        self._test_parametric(F.gelu, gelu, 3)

    def test_hardswish(self):
        self._test_parametric(F.hardswish, hardswish, 3)

    def test_logsigmoid(self):
        self._test_parametric(F.logsigmoid, logsigmoid, 3)

    def test_mish(self):
        self._test_parametric(F.mish, mish, 3)

    def test_selu(self):
        self._test_parametric(F.selu, selu, 3)

    def test_sigmoid(self):
        self._test_parametric(T.sigmoid, sigmoid, 3)

    def test_silu(self):
        self._test_parametric(F.silu, silu, 3)

    def test_softplus(self):
        self._test_parametric(F.softplus, softplus, 3)

    def test_softsign(self):
        self._test_parametric(F.softsign, softsign, 3)

    def test_tanh(self):
        self._test_parametric(T.tanh, tanh, 3)

    def test_tanhshrink(self):
        self._test_parametric(F.tanhshrink, tanhshrink, 3)
