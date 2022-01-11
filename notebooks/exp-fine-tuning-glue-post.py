# -*- coding: utf-8 -*-
# # Few-Bit: Postprocessing of GLUE fine-tuning

import matplotlib.pyplot as plt
import pandas as pd
import tensorboard as tb
import tensorboard.data_compat
import tensorflow as tf

from pathlib import Path
from typing import List, Optional

# First of all, define some globals.

METRICS = {'eval/accuracy', 'eval/matthews_correlation', 'eval/pearson'}
MDASH = '—'


def convert(srcs: List[Path], dst: Optional[Path] = None) -> pd.DataFrame:
    """Function convert reads TensorBoard (written by HuggingFace transformers
    library), filters it contents, and write it to CSV file.
    """
    records = []
    for i, el in enumerate(tf.data.TFRecordDataset(srcs)):
        event = tb.compat.proto.event_pb2.Event.FromString(el.numpy())
        event = tb.data_compat.migrate_event(event)
        if event.summary is None:
            continue
        for value in event.summary.value:
            prefix, _ = value.tag.split('/', 1)
            if prefix not in ('eval', 'train'):
                continue
            records.append((event.step, value.tag, value.tensor.float_val[0]))

    df = pd.DataFrame(data=records, columns=('step', 'tag', 'value'))
    df = df.set_index(['tag', 'step'])
    df = df.sort_index()
    if dst:
        df.to_csv(dst)
    return df


def format_index(x: str) -> str:
    try:
        return f'{int(x)}%'
    except Exception:
        return x.capitalize()


# Find all summary files in specific directory.

dirname = Path('../log')
# filenames = !find "$dirname" -maxdepth 3 -type f -printf "%P\n"

# Read all TensorBoard logs.

frames = []
for filename in filenames:
    param, task, _ = filename.split('/', 2)
    path = dirname / filename
    frame = convert(path)
    frame = frame.reset_index()
    frame['task'] = task.upper()
    frame['param'] = param
    frames.append(frame)

# Select neccesary columns and filter rows by metric name.

df = pd.concat(frames) \
    .set_index(['task', 'param', 'tag', 'step']) \
    .sort_index() \
    .reset_index()
df = df[df.tag.isin(METRICS)]
df.to_csv('summary.csv')

# Make summary table and export it to LaTeX.

summary = df \
    .groupby(['task', 'param'])[['value']] \
    .max()
summary['value'] = summary.value * 100
summary = summary\
        .pivot_table(['value'], ['param'], ['task']) \
        .sort_index(ascending=False)
summary.columns = summary.columns.levels[1]
summary.columns.rename(None, inplace=True)
summary.index = summary.index.map(format_index)
summary.index.rename(None, inplace=True)
summary.head()

summary.to_latex(buf='table.tex',
                 na_rep=f'{MDASH:^5s}',
                 float_format=lambda x: f'{x:5.2f}',
                 caption='Fine-tuning on GLUE tasks.',
                 label='tab:glue-fine-tuning')

# !cat 'table.tex'

df = pd.concat(frames)
df = df[df.tag == 'train/train_samples_per_second'] \
    .set_index(['task', 'param']) \
    .sort_index()
df = df[['value']]

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
ratios = [1, 5, 10, 20, 50, 90, 100]
index = ['01', '05', '10', '20', '50', '90', '100', 'baseline']

# +
fig, ax = plt.subplots(figsize=(6.7, 3), dpi=150)
fig.suptitle('Throughput of model with Gaussian randomized matmul')

for color, (key, group) in zip(colors, df.groupby(level=[0])):
    values = group \
        .reset_index(level=[0], drop=True) \
        .reindex(index=index) \
        .value
    throughput, baseline = values[:-1].values, values[-1]
    ax.plot(ratios, throughput / baseline, '-', color=color, label=key)

ax.grid()
ax.legend()
ax.set_xlim(0, 100)
ax.set_xticks([0, 10, 20, 30, 40, 50, 60, 70])
ax.set_xlabel('Compression ratio, %')
ax.set_ylabel('Speed up in throughput')
