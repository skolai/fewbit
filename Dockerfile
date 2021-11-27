FROM nvcr.io/nvidia/pytorch:21.11-py3

LABEL maintainer "Daniel Bershatsky <daniel.bershatsky@gmail.com>"

# Set up default timezone.

ENV TZ=Europe/Moscow

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && \
    echo $TZ > /etc/timezone

# Set up workspace directory to non-root user.

WORKDIR /workspace/few-bit-backward

ARG LOGIN=developer

ARG UID=1000

RUN useradd -m -U -u $UID $LOGIN && \
    chown -R $LOGIN /workspace && \
    pwd

USER $LOGIN

# Build native library.

ADD --chown=$UID . .

RUN python setup.py build_ext \
    --inplace \
    --cmake-prefix-path "$(python -c 'import torch.utils; print(torch.utils.cmake_prefix_path)')"
