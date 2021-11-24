#   encoding: utf-8
#   filename: __init__.py

import torch as T

from pathlib import Path
from warnings import warn

try:
    T.ops.load_library(Path(__file__).with_name('libfewbit.so'))
except Exception as e:
    warn(f'Failed to load ops library: {e}.', RuntimeWarning)
finally:
    del Path, warn
