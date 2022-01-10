# -*- coding: utf-8 -*-
# # Few-Bit: Postprocessing of GLUE fine-tuning

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
    task, param, _ = filename.split('/', 2)
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
        .pivot_table(['value'], ['task'], ['param']) \
        .sort_index(ascending=False)
summary.columns = summary.columns.levels[1]
summary.index = summary.index.map(format_index)
summary.index.rename(None, inplace=True)
summary.head()
summary.to_latex(buf='table.tex',
                 na_rep=f'{MDASH:^5s}',
                 float_format=lambda x: f'{x:5.2f}',
                 caption='Fine-tuning on GLUE tasks.',
                 label='tab:glue-fine-tuning')

# !cat 'table.tex'
