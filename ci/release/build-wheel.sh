#!/bin/bash
# This scripts is supposed to be run in docker container. It creates and
# activates a virtual env, install core dependencies and build wheel.
#
#   build-wheel.sh <cuda-version> <py-version> <torch-version> <fewbit-version>

CUDA_VERSION=${1:-11.7}
CUDA_VERSION=${CUDA_VERSION::4}
PYTHON_VERSION=${2:-3.10}
[[ -z "$3" ]] && { echo "Torch version is not specified" ; exit 1; }
TORCH_VERSION=$3
FEWBIT_VERSION=${4:-0.0.0}


python$PYTHON_VERSION -m venv .env/py$PYTHON_VERSION/cu$CUDA_VERSION
. .env/py${PYTHON_VERSION}/cu${CUDA_VERSION}/bin/activate

set -xe

pip3 install -U \
    --extra-index-url "https://download.pytorch.org/whl/cu${CUDA_VERSION/./}" \
    "auditwheel" \
    "setuptools" \
    "setuptools_scm>=3.4" \
    "wheel" \
    "numpy" \
    "torch==${TORCH_VERSION}+cu${CUDA_VERSION/./}"

SETUPTOOLS_SCM_PRETEND_VERSION=$FEWBIT_VERSION python3 setup.py \
    build_ext -i --cuda \
    bdist_wheel \
        -d dist/cu${CUDA_VERSION/./}/pt${TORCH_VERSION}/fewbit \
        -p manylinux2014_x86_64

# TODO Add check with auditwheel on manylinux2014 compliance.
# auditwheel show dist/cu$${CUDA_VERSION/./}/fewbit/fewbit-....whl

echo "Make sanity check and show package versions"
python3 - <<-END
import sys
import torch as T
import fewbit
print('Python version is', sys.version)
print('PyTorch version is', T.version.__version__)
print('FewBit version is', fewbit.__version__)
END
