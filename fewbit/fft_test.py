import numpy as np
import scipy as sp
import scipy.fft
import torch as T

from itertools import product
from unittest import TestCase

from fewbit.fft import dct, idct


class TestDCT(TestCase):
    def _test_impl(self, lhs, rhs, delta=1e-5):
        norms = ('backward', 'forward', 'ortho')
        parities = (False, True)
        for parity, norm in product(parities, norms):
            with self.subTest(parity=parity, norm=norm):
                xs = np.random \
                    .RandomState(42) \
                    .randn(37, 2, 127 + int(parity))
                ys = lhs(xs, norm=norm)
                zs = rhs(T.tensor(xs), norm=norm).numpy()
                rerr = np.linalg.norm(zs - ys) / np.linalg.norm(ys)
                self.assertAlmostEqual(0, rerr, delta=delta)

    def test_dct(self):
        self._test_impl(sp.fft.dct, dct)

    def test_idct(self):
        # NOTE Normalization mode backward and forward results in poor
        # accuracy but ortho gets good precision.
        self._test_impl(sp.fft.idct, idct, delta=5e-2)

    def test_identity(self):
        def lhs(xs, *args, **kwargs):
            ys = sp.fft.dct(xs, *args, **kwargs)
            zs = sp.fft.idct(ys, *args, **kwargs)
            return zs

        def rhs(xs, *args, **kwargs):
            ys = dct(xs, *args, **kwargs)
            zs = idct(ys, *args, **kwargs)
            return zs

        # Test identity transformation idct(dct(x)) == identity(x).
        self._test_impl(lhs, rhs, delta=5e-2)
