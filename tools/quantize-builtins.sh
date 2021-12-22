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
                -b "$beps" \
                -l "$leps" \
                -o "$output" \
                -s "$seed" \
                "$nobits" \
                "$module:$func"
        done
    done
}

function build_quantizations_hardswish() {
    for func in ${funcs[@]}; do
        for nobits in {1..4}; do
            echo "make $nobits-bit quantization for gradients of $func"
            PYTHONPATH=$(pwd) tools/quantize-hardswish.py -o "$output" "$nobits"
        done
    done
}

beps=1e-6
leps=1e-6
output=builtin.npz
output=test.npz
seed=42

module=torch
funcs=('sigmoid' 'tanh')
build_quantizations

module=torch.nn.functional
funcs=('celu' 'elu' 'gelu' 'mish' 'silu' 'softplus' 'softsign' 'tanhshrink')
build_quantizations

module=torch.nn.functional
funcs=('logsigmoid' 'selu')
beps=5e-3
leps=1e-1
build_quantizations

echo 'run separate procedure to manually quantize gradients for hardswish'
module=torch.nn.functional
funcs=('hardswish')
build_quantizations_hardswish
