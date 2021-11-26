# few-bit-backward

## Assembly

Preliminary step depends on one's PyTorch distribution and availiable tooling.
Building of native components requires CMake and a build system like Make or Ninja.
Next, if PyTorch is installed system-wide the the following step is not neccessary.
Otherwise, one likely should add search path for CMake modules to environment variables as follows.

```
export CMAKE_PREFIX_PATH="$(python -c 'import torch.utils; print(torch.utils.cmake_prefix_path)')"
```

The next step is useful in development environment.
It just builds PyTorch operator library in source tree (option `--inplace`) with forced CUDA support (option `--cuda`).
By default no CUDA support are forced.

```bash
python setup.py build_ext --inplace --cuda
```

With options similar to the previous step, one can build wheel binary distribution of the package.

```bash
python setup.py bdist_wheel --inplace --cuda
```
