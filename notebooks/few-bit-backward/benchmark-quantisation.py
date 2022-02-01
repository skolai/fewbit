# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.0
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# + id="fLuHW0FhrVqC"
import torch as T
import torch.utils.benchmark as benchmark

# + id="OfSpSFdhsPMl"
device = T.device('cuda')

# + id="z6qUKC39rzwR"
bounds = T.tensor([
    -2.39798704e+00,
    -7.11248159e-01,
    -3.26290283e-01,
    -1.55338428e-04,
    3.26182064e-01,
    7.10855860e-01,
    2.39811567e+00,
]).to(device)


# @T.jit.script
def forward1(xs, bounds):
    return T.searchsorted(bounds, xs).type(T.uint8)


# @T.jit.script
def forward2(xs, bounds):
    value, pos = T.min(xs[..., None] > bounds, -1)
    discr = T.where(value, len(bounds), pos).type(T.uint8)
    return discr


# + id="3bB41SCWr9LF"
xs = T.randn((128, 128)).to(device)

# + id="_b0skuBgsZgE"
# Warm up JIT and make sanity check
y1 = forward1(xs, bounds)
y2 = forward2(xs, bounds)
assert T.all(y1 == y2)

# + id="997lStURsJn2" outputId="6b980cd5-66d8-4444-85d9-2b4cadfa2c24"
# %timeit forward1(xs, bounds)
# %timeit forward2(xs, bounds)

# + id="i0ckMfwxvC_L"
t1 = benchmark.Timer(
    stmt='forward1(xs, bounds)',
    setup='from __main__ import forward1',
    globals={'xs': xs, 'bounds': bounds})

t2 = benchmark.Timer(
    stmt='forward2(xs, bounds)',
    setup='from __main__ import forward2',
    globals={'xs': xs, 'bounds': bounds})

# + id="xpCvWpAHvOsU" outputId="4acfc3a1-7c2e-4429-b821-14fcf631b573"
print(t1.timeit(10000))
print(t2.timeit(10000))
