#   encoding: utf-8
#   filename: approx.py

import numpy as np
import scipy as sp
import scipy.integrate

from io import StringIO
from typing import Any, Callable, Dict, Tuple, Union

from numpy.typing import ArrayLike, NDArray

__all__ = ('StepWiseFunction', 'approximate', 'estimate_error', 'stepwise')

RandomState = Union[None, int, ArrayLike, np.random.RandomState]
VectorizedFn = Callable[[ArrayLike], NDArray]


class StepWiseFunction:
    """Class StepWiseFunction implements piecewise constant function.

    :param borders:
    :param levels:
    """

    def __init__(self, borders, levels):
        assert borders.ndim == 1
        assert levels.ndim == 1
        assert borders.size == levels.size + 1

        self.borders = borders
        self.card = self.borders.size - 1
        self.levels = levels
        self.steps = levels.copy()
        self.steps[1:] = levels[1:] - levels[:-1]

    def __call__(self, xs):
        spikes = np.heaviside(xs[None, ...] - self.borders[1:-1, None], 0.0)
        values = self.steps[0, None] + self.steps[1:] @ spikes
        return values

    def __repr__(self) -> str:
        borders = ', '.join([f'{x:e}' for x in self.borders[:2]])
        levels = ', '.join([f'{x:e}' for x in self.levels[:2]])
        if self.borders.size > 2:
            borders += ', ...'
            levels += ', ...'
        return (f'<StepWiseFunction nosteps={self.levels.size} '
                f'borders=[{borders}] levels=[{levels}]>')

    def __str__(self) -> str:
        buf = StringIO()
        for i in range(self.card):
            a, b = self.borders[i:i + 2]
            level = self.levels[i]
            print(f'[{i}] [{a:+8.3f}, {b:+8.3f}) => {level:e}', file=buf)
        return buf.getvalue()


def stepwise(xs, ys) -> StepWiseFunction:
    return StepWiseFunction(xs, ys)


def approximate(
    fn: VectorizedFn,
    fn_prim: VectorizedFn,
    cardinality: int,
    domain: Tuple[float, float] = (-100.0, 100.0),
    parity: bool = False,
    max_iters: int = 10000,
    beps: float = 1e-4,
    leps: float = 1e-4,
    random_state: RandomState = None
) -> Tuple[StepWiseFunction, Dict[str, Any]]:
    """Function approximate builds a stepwise function approximation on a
    specified domain with given parity property.

    :param fn: Function.
    :param fn_prim: Primitive function
    :param cardinality:
    :param domain:
    :param parity:
    :param max_iters:
    :param beps:
    :param leps:
    :param random_state:

    :return: Tuple of StepWiseFunction object and dictionary object with
             convergence status.
    """
    x_min, x_max = domain
    if parity:
        assert x_min == 0.0
    rng = np.random.RandomState(random_state)

    # Generate initial guess for function approximation.
    def initializer(rng, size):
        bs = rng.normal(0.0, 1.5, size)
        if parity:
            bs = np.abs(bs)
        return bs

    # Initialize interval borders.
    bs = np.empty(cardinality + 1)
    bs[0] = x_min
    bs[-1] = x_max
    for _ in range(16):
        bs[1:-1] = initializer(rng, cardinality - 1)
        bs.sort()
        if is_sorted(bs, 1e-3):
            break
    else:
        raise RuntimeError('Failed to generate initial lattice!')

    # Initialize function values on pieces.
    ls = np.diff(fn_prim(bs)) / np.diff(bs)

    # Run two-step optimisation procedure.
    for cur_iter in range(max_iters):
        # Update interval borders.
        ys_diff = np.diff(ls)
        ys_mean = 0.5 * (ls[:-1] + ls[1:])
        bs_grad = -2 * ys_diff * (fn(bs[1:-1]) - ys_mean)
        bs[1:-1] += bs_grad
        bs_diff = np.linalg.norm(bs_grad) / np.linalg.norm(bs)

        if np.linalg.norm(bs_grad) < beps:
            status = 'converged'
            break

        # Reestimate levels.
        ls_next = np.diff(fn_prim(bs)) / np.diff(bs)
        ls_diff = np.linalg.norm(ls_next - ls) / np.linalg.norm(ls)
        ls = ls_next

        # Check stopping criterion for levels.
        if ls_diff < leps:
            status = 'converged'
            break

        # Assume that all interval borders are sorted. Violation of order
        # results in convergence issues.
        if not is_sorted(bs):
            status = 'failed'
            break
    else:
        status = 'not-converged'

    return stepwise(bs, ls), {
        'status': status,
        'noiters': cur_iter,
        'bs_diff': bs_diff,
        'ls_diff': ls_diff,
    }


def estimate_error(fn, fn_approx, dx):
    es = np.empty(fn_approx.card + 1)
    for i in range(fn_approx.card):
        a, b = fn_approx.borders[i:i + 2]
        nopoints = min(1024 ** 2, int((b - a) / dx))
        xs = np.linspace(a, b, nopoints)
        ys = fn_approx.levels[i]
        es[i] = sp.integrate.simpson((fn(xs) - ys)**2, xs, dx)
    return es.sum(), es


def is_sorted(xs: ArrayLike, scale: float = 0.0) -> bool:
    return np.all(np.diff(xs) > scale)
