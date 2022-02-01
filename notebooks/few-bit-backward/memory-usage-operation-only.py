# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Memory Usage: Operator Only

import pandas as pd
import torch as T

from sys import path
path.insert(0, '..')
import fewbit  # noqa

# Define parameters of quantisation.

# +
BOUNDS = T.tensor([
    -2.39798704e+00, -7.11248159e-01, -3.26290283e-01, -1.55338428e-04,
    3.26182064e-01, 7.10855860e-01, 2.39811567e+00
], dtype=T.float32, device=T.device('cuda'))

LEVELS = T.tensor([
    -0.00260009, -0.08883533, 0.1251944, 0.37204148, 0.6277958, 0.87466175,
    1.08880716, 1.00259936
], dtype=T.float32, device=T.device('cuda'))

# -

# Preallocate inputs and output gradients.

xs = T.zeros(128 * 1024 ** 2, device=T.device('cuda'), requires_grad=True)
gs = T.ones_like(xs)

# Execute operators under profiler.

# +
with T.profiler.profile(
    schedule=T.profiler.schedule(wait=1, warmup=1, active=2, repeat=1),
    on_trace_ready=T.profiler.tensorboard_trace_handler('../log/op-only'),
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    prof.step()
    for _ in range(2):
        ys = T.nn.functional.gelu(xs)
        ys.backward(gs)
        print(xs.grad[0], xs.grad[-1])
        prof.step()

with T.profiler.profile(
    schedule=T.profiler.schedule(wait=1, warmup=1, active=2, repeat=1),
    on_trace_ready=T.profiler.tensorboard_trace_handler('../log/op-only-opt'),
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    prof.step()
    for _ in range(2):
        ys = T.ops.fewbit.gelu(xs, BOUNDS, LEVELS)
        ys.backward(gs)
        print(xs.grad[0], xs.grad[-1])
        prof.step()
# -

# Estimate computational speed up.
# (We take values from `Operator` view of `torch-tb-profiler`.)

# +
df = pd.DataFrame([
    ('forward', 'original', 1.314),
    ('forward', 'quantized', 2.862),
    ('backward', 'original', 1.906),
    ('backward', 'quantized', 1.563),
], columns=('pass', 'impl', 'elapsed'))

df_total = df \
    .set_index('impl') \
    .groupby(level=0) \
    .agg({'pass': 'first', 'elapsed': 'sum'}) \
    .reset_index()
df_total['pass'] = 'total'

df = df \
    .append(df_total, True) \
    .set_index(['impl', 'pass']) \
    .sort_index()

# +
speedup = df.elapsed.values.copy()
speedup[:3] = speedup[:3] / df.elapsed.values[:3]
speedup[3:] = speedup[3:] / df.elapsed.values[:3]

df['speedup'] = speedup
df['diff'] = df.speedup - 1
# -

with pd.option_context('display.float_format', '{:,.2f}'.format):
    print(df)
