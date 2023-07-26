# FewBit

**FewBit** &mdash; a library for memory efficient training of large neural networks.
Its efficiency originates from storage optimizations applied to backward pass and memory footprint reduction for saved tensors between forward and backward passes.
Namely, the library provides its own implementation of common activation functions and linear layer since they contribute the most to memory usage in training time.
Optimized linear layer saves up to 15-20% memory and optimized activation functions save up to 15-30% of memory usage with negligible loss in performance (see \[[1][5]\]\[[2][6]\] for details).

In the table below, one can see comparison of different optimizations applied to RoBERTa model. Compression rate of randomized linear layer is 20% (it uses only 20% of input) and GELU approximation uses only 3 bits.

<table>
  <thead>
    <tr style="text-align: right;">
      <th></th><th>Task</th><th>Batch Size</th><th>GELU</th><th>Linear Layer</th><th>Peak Memory, GiB</th><th>Saving, %</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th><td>MRPC</td><td>128</td><td>Vanilla</td><td>Vanilla</td><td>11.30</td><td>0.0</td>
    </tr>
    <tr>
      <th>2</th><td>MRPC</td><td>128</td><td>3-bit</td><td>Vanilla</td><td>9.75</td><td>13.8</td>
    </tr>
    <tr>
      <th>3</th><td>MRPC</td><td>128</td><td>Vanilla</td><td>Randomized</td><td>9.20</td><td>18.6</td>
    </tr>
    <tr>
      <th>4</th><td>MRPC</td><td>128</td><td>3-bit</td><td>Randomized</td><td>7.60</td><td>32.7</td>
    </tr>
  </tbody>
</table>

## Usage

The library `fewbit` implements basic activation functions with backward pass
optimizations for reducing memory footprint during model training.
All activation functions exported by the library can be used as a drop-in
replacement for most of standard activation functions implemented in PyTorch.
The common pattern is to replace `torch.nn` with `fewbit` package qualifier.

```python
import fewbit
import torch as T

model = T.nn.Sequential(
    ...,
    fewbit.GELU(bits=3),  # Use 3-bits GELU approximation.
    ...,
)
```

In the case of pre-trained models, one can rebuild model with `map_module` routine which walks through model tree recursively and allows to replace some modules or activation functions.
So, user should only use suitable constructor for a new module.
As an example the code below replaces all default linear layers with randomized ones.

```python
from fewbit import RandomizedLinear
from fewbit.util import convert_linear, map_module

converter = lambda x: convert_linear(x, RandomizedLinear, proj_dim_ratio=0.1)
new_model = map_module(old_model, converter)  # In-place model construction.
```

![Quantized Gradients of Activation Functions][4]

### Installation

The simplest and preferred installation way is installation from PyPI.

```shell
pip install -U fewbit
```

FewBit is written in Python, but it implements some opertions in C++/CUDA to archive better performance.
So, building from source requires CUDA Toolkit and CMake as a build system.
The latest release can be installed with the following command.

```shell
pip install -U https://github.com/SkoltechAI/fewbit.git
```

Another one way to get FewBit is an installation from pre-built wheels from
custom PyPI. Assume that CUDA version is 11.7 and desired PyTorch version is
2.0.1 then the command below downloads and installes PyTorch of specified
version and the latest availiable FewBit.

```shell
pip install fewbit torch==2.0.1 \
    --extra-index-url https://download.pytorch.org/whl/cu117 \
    --extra-index-url https://mirror.daskol.xyz/pypi/cu117/pt2.0.1
```

Note that URLs of the custom PyPIs are built from CUDA version and PyTorch
version and can be manually adjusted (see [this page][7] for list of pre-built
wheels).

### List of Activation Functions

The library supports the following activation functions.

#### Piece-wise Activation Functions

In this section, all activation functions has 1-bit derivative.
The only difference is band.
The band requires two comparison to determine gradient domain.
The complete list of activation functions is `leaky_relu`, `relu`,
`threshold`, `hardsigmoid`, `hardtanh`, `relu6`, `hardshrink`, and
`softshrink`.

##### Continous Activation Functions

All continous activation function could be divided into three classes according to its parity property: odd, even, and neither even nor odd.
The parity property allows to use a small optimization to increase precision of approximation.
The complete list of reimplemented activation functions in this category is
`celu`, `elu`, `hardswish`, `logsigmoid`, `mish`, `selu`, `sigmoid`, `silu`,
`softplus`, `softsign`, `tanh`, and `tanhshrink`.

### List of Modules

Module `RandomizedLinear` is a replacement for default `Linear` module.
It is used power of approximate matrix multiplication for memory saving.

## Assembly

Preliminary step depends on one's PyTorch distribution and availiable tooling.
Building of native components requires CMake and a build system like Make or Ninja.
Next, if PyTorch is installed system-wide the the following step is not neccessary.
Otherwise, one likely should add search path for CMake modules to environment variables as follows.

```shell
export CMAKE_PREFIX_PATH="$(python -c 'import torch.utils; print(torch.utils.cmake_prefix_path)')"
```

The next step is useful in development environment.
It just builds PyTorch operator library in source tree (option `--inplace`) with forced CUDA support (option `--cuda`).
By default no CUDA support are forced.

```shell
python setup.py build_ext --inplace --cuda
```

With options similar to the previous step, one can build wheel binary distribution of the package.

```shell
python setup.py bdist_wheel --inplace --cuda
```

## Development Environment with Docker

In order to develop on different platforms we uses custom docker image for non-priviledge user based on Nvidia CUDA image.
Image contains pre-built native extention and it is parametrized by user name and user ID in a host system.
The latter is crucial thing in binding host volumes.

```shell
docker build -t fewbit --build-arg UID=$(id -u) .
docker run --rm -ti -e TERM=$TERM fewbit
```

## Citation

Please cite the following papers if the library is used in an academic paper (export [BibTeX][1]).

```bibtex
@misc{bershatsky2022memoryefficient,
    title={{M}emory-{E}fficient {B}ackpropagation through {L}arge {L}inear {L}ayers},
    author={Daniel Bershatsky and Aleksandr Mikhalev and Alexandr Katrutsa and Julia Gusak and Daniil Merkulov and Ivan Oseledets},
    year={2022},
    eprint={2201.13195},
    archivePrefix={arXiv},
    primaryClass={cs.LG},
}

@misc{novikov2022fewbit,
    title={{F}ew-{B}it {B}ackward: {Q}uantized {G}radients of {A}ctivation {F}unctions for {M}emory {F}ootprint {R}eduction},
    author={Georgii Novikov and Daniel Bershatsky and Julia Gusak and Alex Shonenkov and Denis Dimitrov and Ivan Oseledets},
    year={2022},
    eprint={2202.00441},
    archivePrefix={arXiv},
    primaryClass={cs.LG},
}
```

## License

© The FewBit authors, 2022 &mdash; now. Licensed under the BSD 3-Clause License. See [AUTHORS][2] and [LICENSE][2] file for more details[^1].

[^1]: The work was supported by Sber AI and the Analytical center under the RF Government (subsidy agreement 000000D730321P5Q0002, Grant No. 70-2021-00145 02.11.2021).

[1]: doc/fewbit.bib
[2]: AUTHORS
[3]: LICENSE
[4]: doc/fig/activations.svg
[5]: https://arxiv.org/abs/2201.13195
[6]: https://arxiv.org/abs/2202.00441
[7]: https://mirror.daskol.xyz/pypi/
