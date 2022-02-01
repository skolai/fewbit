# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %load_ext autoreload
# %autoreload 2
# %env CUDA_VISIBLE_DEVICES=7

import torch as t
from torch import nn
import numpy as np
import scipy
import scipy.special
import scipy.optimize
import scipy.stats
from matplotlib import pyplot as plt
from tqdm.auto import trange, tqdm


# +
def gelu(x):
    return .5 * x * (1 + scipy.special.erf(x / np.sqrt(2)))


def gelu_deriv(x):
    return .5 * scipy.special.erf(x / np.sqrt(2)) + x * np.exp(-x**2 / 2) / np.sqrt(2 * np.pi) + .5


# -

# # Approximation

# ### Dynamic Programming

# +
def calc_y_opt(int_w, int_f_w):
    num = int_f_w[None, :] - int_f_w[:, None]

    den = int_w[None, :] - int_w[:, None]
    np.fill_diagonal(den, 1.)
    den[den == 0] = 1.

    return num / den


def calc_trainsition_matrix(int_w, int_f2_w, y_opt):
    translation_matrix = int_f2_w[None, :] - int_f2_w[:, None] - y_opt**2 * (int_w[None, :] - int_w[:, None])
    translation_matrix = np.where(np.tri(len(int_w), k=-1, dtype=np.bool).T, translation_matrix, np.inf)
    return translation_matrix
    
    
def restore_answer(frm):
    current = len(frm[-1]) - 1
    points = [current]
    for frm_curr in frm[::-1]:
        current = frm_curr[current]
        points.append(current)
    points = points[::-1]
    
    assert points[0] == 0
    assert points[-1] == len(frm[-1]) - 1
    
    return points


def dynamic_programming(int_w, int_f_w, int_f2_w, xs, k: int):
    y_opt = calc_y_opt(int_w, int_f_w)
    transition_matrix = calc_trainsition_matrix(int_w, int_f2_w, y_opt)

    def one_step(dp):
        dp_new = dp[:, None] + transition_matrix
        frm = np.argmin(dp_new, axis=0)
        return np.min(dp_new, axis=0), frm

    dp = np.full(len(int_w), np.inf)
    dp[0] = 0
    frm = []

    for _ in trange(k):
        dp, from_curr = one_step(dp)
        frm.append(from_curr)

    points = restore_answer(frm)
    
    return xs[points], y_opt[points[:-1], points[1:]]


def numerical_integral(f_values, delta_x):
    return np.cumsum(f_values) * delta_x


def uniform_weight(n_segments, integral_n_coeff, L: float, R: float, n_bits: int, act, act_deriv):
    """
    \int_L^R (act_deriv(x) - q(x | xs, ys))^2 dx -> min
    
    n_segments: number of discretization points
    integral_n_coeff: how many points to use inside each interval to perform numerical integration
    L: left bound 
    R: right bound
    n_bits: number of bits in piecewise-constant approximation (2**n_bits is number of intervals)
    act: callable representing activation function
    act_deriv: callable representing derivative of activation function
    """
    
    xs = np.linspace(L, R, n_segments + 1)

    int_w = xs - xs[0]
    int_f_w = act(xs) - act(xs[0])

    xs_int = np.linspace(L, R, (n_segments + 1) * (integral_n_coeff + 1) - integral_n_coeff)
    int_f2_w = numerical_integral(act_deriv(xs_int)**2, xs_int[1] - xs_int[0])[::integral_n_coeff + 1]
    
    return dynamic_programming(int_w, int_f_w, int_f2_w, xs, 2**n_bits)


def all_numerical(n_segments, integral_n_coeff, L: float, R: float, n_bits: int, act_deriv, w):
    """
    int (act_deriv(x) - q(x | xs, ys))^2 w(x) dx -> min
    
    
    n_segments: number of discretization points
    integral_n_coeff: how many points to use inside each interval to perform numerical integration
    L: left bound 
    R: right bound
    n_bits: number of bits in piecewise-constant approximation (2**n_bits is number of intervals)
    act: callable representing activation function
    act_deriv: callable representing derivative of activation function
    """
    
    xs = np.linspace(L, R, n_segments + 1)
    xs_int = np.linspace(L, R, (n_segments + 1) * (integral_n_coeff + 1) - integral_n_coeff)
    delta_x = xs_int[1] - xs_int[0]
    
    int_w = numerical_integral(w(xs_int), delta_x)[::integral_n_coeff + 1]
    int_f_w = numerical_integral(act_deriv(xs_int) * w(xs_int), delta_x)[::integral_n_coeff + 1]
    int_f2_w = numerical_integral(act_deriv(xs_int)**2 * w(xs_int), delta_x)[::integral_n_coeff + 1]
    
    return dynamic_programming(int_w, int_f_w, int_f2_w, xs, 2**n_bits)


N_SEGMENTS = 2**12
INTEGRAL_N_COEFF = 2**10
L = -10
R = 10
N_BITS = 5

xs, ys = uniform_weight(N_SEGMENTS, INTEGRAL_N_COEFF, L, R, N_BITS, gelu, gelu_deriv)
# xs, ys = all_numerical(N_SEGMENTS, INTEGRAL_N_COEFF, L, R, N_BITS, gelu_deriv, lambda x: scipy.stats.norm.pdf(x * .5))

plt.plot(np.linspace(L, R, 2**10), gelu_deriv(np.linspace(L, R, 2**10)))
for x1, x2, y in zip(xs[:-1], xs[1:], ys):
    plt.plot([x1, x2], [y, y])
