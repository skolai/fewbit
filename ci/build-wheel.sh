#!/bin/bash
#
#   Script for building binary distribution (wheel) in isolated environment
#   (docker container).
#
#       ci/build-wheel.sh 11.5 3.10 1.10
#

CUDA_VERSION=$1
CUDA_VERSION_SHORT=$(cut -f 1,2 -d . <<< $CUDA_VERSION)
CUDA_ORDINAL=$(tr -d '.' <<< $CUDA_VERSION_SHORT)

PYTHON_VERSION=$2
PYTHON="python${PYTHON_VERSION}"
PYTHON_ABI="cp$(tr -d '.' <<< $PYTHON_VERSION)"
PYTHON_PREFIX="/opt/python/${PYTHON_ABI}-${PYTHON_ABI}"

TORCH_VERSION=$3

echo "Build wheel in context of"
echo "    CUDA          $CUDA_VERSION"
echo "    Python        $PYTHON_VERSION"
echo "    Torch         $TORCH_VERSION"

PATH="${PYTHON_PREFIX}/bin:$PATH"
PYPI_EXTRA_INDEX="https://download.pytorch.org/whl/cu${CUDA_ORDINAL}/"

# Ad hoc solution to fix git repo.
git config --global --add safe.directory /workspace

$PYTHON -m pip install -U --extra-index-url "$PYPI_EXTRA_INDEX" \
    "auditwheel" \
    "setuptools" \
    "setuptools_scm>=3.4" \
    "wheel" \
    "numpy" \
    "torch==${TORCH_VERSION}+cu${CUDA_ORDINAL}"

$PYTHON setup.py \
    build_ext -i --cuda \
    bdist_wheel \
        -d dist/cu$CUDA_ORDINAL/fewbit \
        -p manylinux2014_x86_64

# TODO Add check with auditwheel on manylinux2014 compliance.
# auditwheel show dist/cu$CUDA_ORDINAL/fewbit/fewbit-....whl

echo "Make sanity check and show package versions"
$PYTHON - <<-END
import sys
import torch as T
import fewbit
print('Python version is', sys.version)
print('PyTorch version is', T.version.__version__)
print('FewBit version is', fewbit.__version__)
END
