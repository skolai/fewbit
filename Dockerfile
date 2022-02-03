FROM nvcr.io/nvidia/pytorch:21.11-py3

LABEL maintainer "Daniel Bershatsky <daniel.bershatsky@gmail.com>"

# Set up default timezone.

ENV TZ=Europe/Moscow

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && \
    echo $TZ > /etc/timezone

# Set up workspace directory to non-root user.

WORKDIR /workspace/fewbit

ARG LOGIN=developer

ARG UID=1000

RUN useradd -m -U -u $UID $LOGIN && \
    chown -R $LOGIN /workspace && \
    pwd

USER $LOGIN

# Build native library.

ARG USE_CUDA=ON

ADD --chown=$UID . .

RUN export CMAKE_PREFIX_PATH=$(python -c 'import torch.utils; print(torch.utils.cmake_prefix_path)'); \
    if [[ "$USE_CUDA" == "ON" ]]; then \
        python setup.py build_ext -i --cuda; \
    else \
        python setup.py build_ext -i; \
    fi
