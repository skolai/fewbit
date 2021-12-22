#!/bin/bash
#
# This script is aimed to build quantization for common activation functions
# and store their parameters in npz-formatted file.

function build_quantizations() {
    for func in ${funcs[@]}; do
        for nobits in {1..4}; do
            echo "make $nobits-bit quantization for gradients of $func"
            python -m fewbit --log-level info quantize \
                -M 100000 \
                -o "$output" \
                -s "$seed" \
                "$nobits" \
                "$module:$func"
        done
    done
}

output=builtin.npz
seed=42

module=torch.nn.functional
funcs=('celu' 'elu' 'gelu' 'hardswish' 'logsigmoid' 'mish' 'selu' 'silu'
       'softplus' 'softsign' 'tanhshrink')
build_quantizations

module=torch
funcs=('sigmoid' 'tanh')
build_quantizations
