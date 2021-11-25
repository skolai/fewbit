#   encoding: utf-8
#   filename: tensorboard.py

import numpy as np
import tensorflow as tf

from contextlib import contextmanager
from datetime import datetime
from itertools import count
from pathlib import Path

from tensorflow.core.util import event_pb2
from tensorflow.python.summary.writer.event_file_writer import EventFileWriter

__all__ = ('TensorBoard', 'tensorboard')


class TensorBoard:

    def __init__(self, logdir):
        if tf.io.gfile.isdir(logdir):
            tf.io.gfile.makedirs(logdir)
        self.sink = EventFileWriter(logdir)
        self.step = count()
        self.closed = False

    def add_summary(self, summary, step):
        if not self.closed:
            event = event_pb2.Event(summary=summary)
            event.wall_time = datetime.now().timestamp()
            if step:
                event.step = step
            self.sink.add_event(event)

    def close(self):
        if not self.closed:
            self.sink.flush()
            self.sink.close()
            self.closed = True
            del self.sink
            self.sink = None

    def scalar(self, tag, value, step=None):
        value = float(np.array(value))
        summary_value = tf.compat.v1.Summary.Value(tag=tag, simple_value=value)
        summary = tf.compat.v1.Summary(value=[summary_value])
        self.add_summary(summary, step or next(self.step))


@contextmanager
def tensorboard(logdir: str) -> TensorBoard:
    tb = TensorBoard(Path(logdir))
    yield tb
    tb.close()
