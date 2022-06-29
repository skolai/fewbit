#   encoding: utf-8
#   filename: __init__.py
"""Package fewbit provides a plenty optimized primitives for the bottleneck
layers of modern neural networks.
"""

import torch as T

from pathlib import Path
from os import getenv
from warnings import warn

# This is feature toggle which enables or disable usage of native
# implementation or primitive operations. We assume that environment variable
# FEWBIT_NATIVE manages loading of native extension. If native extension is not
# loaded then fallback implementation is used.
if getenv('FEWBIT_NATIVE') not in ('0', 'no', 'false'):
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
