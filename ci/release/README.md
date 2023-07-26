# FewBit: Release

Assume we want to build Python wheels for `fewbit` for CUDA 11.7. In this case
we should find the latest Docker image with such major and minor version (e.g.
nvidia/cuda:11.7.2-devel-ubuntu20.04). Then we should run the command below
with CUDA vesion in build argument. This and all other commands are executed
from repo root.

```shell
docker build \
    -f ci/release/Dockerfile \
    -t github.com/skoltech-ai/fewbit/sandbox:cu117 \
    --build-arg CUDA_VERSION=11.7.2 .
```

As soon as builder image is ready we can build a wheel for specific versions of
Python and Torch.

```shell
docker run --rm -ti \
    -v $PWD:/usr/src/fewbit \
    -w /usr/src/fewbit \
    github.com/skoltech-ai/fewbit/sandbox:cu118 \
    ci/release/build-wheel.sh 11.8 3.10 2.0.0 0.1.0
```
