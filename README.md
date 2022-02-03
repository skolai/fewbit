# FewBit

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

### List of Activation Functions

The library only supports element-wise activation functions.

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
docker build -t few-bit-backward --build-arg UID=$(id -u) .
docker run --rm -ti -e TERM=$TERM few-bit-backward
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
