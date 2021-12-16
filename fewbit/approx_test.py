import numpy as np
import scipy as sp

from unittest import TestCase

from fewbit.approx import approximate


def gelu(x):
    return 0.5 * x * (1 + sp.special.erf(x / np.sqrt(2)))


def gelu_grad(x):
    return (
        0.5 * (1 + sp.special.erf(x / np.sqrt(2))) +
        x * np.exp(- 0.5 * x ** 2) / np.sqrt(2 * np.pi)
    )


class TestApproximate(TestCase):

    def setUp(self):
        self.borders = np.array([
            -2.39798704e+00, -7.11248159e-01, -3.26290283e-01, -1.55338428e-04,
            3.26182064e-01, 7.10855860e-01, 2.39811567e+00
        ])

        self.levels = np.array([
            -0.00260009, -0.08883533, 0.1251944, 0.37204148, 0.6277958,
            0.87466175, 1.08880716, 1.00259936
        ])

        self.kwargs = {
            'fn': gelu_grad,
            'fn_prim': gelu,
            'cardinality': 2**3,
            'parity': False,
            'max_iters': 2000,
            'beps': 1e-6,
            'leps': 1e-6,
            'domain': (-100, 100),
            'random_state': 42,
        }

    def test_approximate(self):
        fn, info = approximate(**self.kwargs)
        self.assertEqual('converged', info['status'])

        err_borders = np.linalg.norm(fn.borders[1:-1] - self.borders)
        self.assertAlmostEqual(0, err_borders, places=1)

        err_levels = np.linalg.norm(fn.levels - self.levels)
        self.assertAlmostEqual(0, err_levels, places=2)

    def test_approximate_parity(self):
        kwargs = {
            **self.kwargs,
            'cardinality': 1 << 2,
            'parity': True,
            'domain': (0, 100),
        }
        fn, info = approximate(**kwargs)
        self.assertEqual('converged', info['status'])

        err_borders = np.linalg.norm(fn.borders[:-1] - self.borders[3:])
        self.assertAlmostEqual(0, err_borders, places=1)

        err_levels = np.linalg.norm(fn.levels - self.levels[4:])
        self.assertAlmostEqual(0, err_levels, places=2)
