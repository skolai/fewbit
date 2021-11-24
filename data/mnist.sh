#!/bin/bash

DATA_DIR=${1:-mnist}

echo "Download MNIST dataset to $DATA_DIR directory."
mkdir -p $DATA_DIR
wget -c -q --show-progress -P "$DATA_DIR" \
    "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz" \
    "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz" \
    "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz" \
    "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"
