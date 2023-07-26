ARG CUDA_VERSION=11.7.1

FROM nvidia/cuda:$CUDA_VERSION-devel-ubuntu20.04

LABEL maintainer "Daniel Bershatsky <d.bershatsky2@skoltech.ru>"

ENV TZ=Europe/Moscow

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && \
    echo $TZ > /etc/timezone

RUN DEBIAN_FRONTEND=noninteractive && \
    apt update && \
    apt install -y --no-install-recommends software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt update && \
    apt install -y --no-install-recommends \
        python3.8 \
        python3.8-dev \
        python3.8-venv  \
        python3.9 \
        python3.9-dev \
        python3.9-venv  \
        python3.10 \
        python3.10-dev \
        python3.10-venv  \
        python3.11 \
        python3.11-dev \
        python3.11-venv
