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
    del Path, T, warn

from . import functional  # noqa: F401
from .modules import *  # noqa: F401,F403
from .util import map_module  # noqa: F401

try:
    from .version import version as __version__
except ImportError:
    __version__ = None
