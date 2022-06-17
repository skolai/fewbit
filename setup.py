from os import environ, makedirs
from pathlib import Path
from shutil import rmtree
from subprocess import check_output
from sys import executable

from packaging.version import parse as parse_version
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext as build_ext_base
from setuptools_scm import dump_version, get_version


class CMakeExtension(Extension):

    def __init__(self, name):
        super().__init__(name, sources=[])


# We needs to name commands like this since introspection is used to print
# usage message.
class build_ext(build_ext_base):

    user_options = build_ext_base.user_options + [
        ('cuda', None, "build with CUDA support"),
        ('cuda-arch=', None,
         "list of CUDA architectures to generate PTX code"),
        ('cmake-prefix-path=', None,
         "semicolon-separated list of directories specifying search prefixes"),
        ('cmake-generator=', None, "supply CMake generator"),
    ]

    boolean_options = build_ext_base.boolean_options + ['cuda']

    def initialize_options(self):
        super().initialize_options()
        self.cmake_generator = None
        self.cmake_prefix_path = None
        self.cuda = False
        self.cuda_arch = 'common'

    def run(self):
        cmake_extensions = []
        rest_extensions = []
        for ext in self.extensions:
            if isinstance(ext, CMakeExtension):
                cmake_extensions.append(ext)
            else:
                rest_extensions.append(ext)
        self.extensions = rest_extensions
        super().run()
        for ext in cmake_extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        build_dir = self.build_temp
        install_dir = self.build_lib
        source_dir = '.'
        if self.inplace:
            environ['CMAKE_INSTALL_MODE'] = 'SYMLINK_OR_COPY'
            install_dir = Path().cwd().absolute()

        build_type = 'RelWithDebugInfo'
        if self.debug:
            build_type = 'Debug'

        # Do nothing on dry run.
        if self.dry_run:
            return

        # If we are forced to rebuild then remove build directory run as usual.
        if self.force:
            rmtree(build_dir)
        makedirs(build_dir, exist_ok=True)

        # Obtain CMake prefix path to PyTorch scripts.
        if not self.cmake_prefix_path:
            self.cmake_prefix_path = get_torch_cmake_prefix_path()

        # Generate project build system.
        cmd = (f'cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -S {source_dir} '
               f'-B {build_dir} -DCMAKE_BUILD_TYPE={build_type}').split()
        if self.cmake_generator:
            cmd.extend(('-G', self.cmake_generator))
        if self.cmake_prefix_path:
            cmd.append(f'-DCMAKE_PREFIX_PATH={self.cmake_prefix_path}')
        if self.cuda:
            cmd.append('-DUSE_CUDA=ON')
        if self.cuda and self.cuda_arch:
            cmd.append(f'-DTORCH_CUDA_ARCH_LIST={self.cuda_arch.capitalize()}')
        if self.debug:
            cmd.append('-DCMAKE_BUILD_TYPE=Debug')
        self.spawn(cmd)

        # Build project.
        cmd = f'cmake --build {self.build_temp}'.split()
        if self.parallel:
            cmd.extend(('-j', str(self.parallel)))
        self.spawn(cmd)

        # Install project to build directory.
        cmd = (f'cmake --install {self.build_temp} '
               f'--prefix {install_dir}').split()
        self.spawn(cmd)


def get_torch_attr(script):
    # We do not want import torch directly in order to save 200+Mb of memory
    # during building extension.
    command = [executable, '-c', script]
    output = check_output(command, encoding='utf-8', timeout=60)
    return output

def get_torch_cmake_prefix_path():
    script = 'import torch.utils; print(torch.utils.cmake_prefix_path)'
    return get_torch_attr(script)


def get_torch_version():
    script = 'import torch as T; print(T.version.__version__)'
    return get_torch_attr(script)


# Get FewBit version and Torch version.
fewbit_version = parse_version(get_version())
torch_version = parse_version(get_torch_version())

# FewBit version is <torch-public>.<fewbit-public>[+<torch-local>] version.
version = '.'.join([torch_version.public, fewbit_version.base_version])
if torch_version.local:
    version += f'+{torch_version.local}'

# Write FewBit version to file.
dump_version('.', version, 'fewbit/version.py')

# We fix Torch version in order to maintain compatibility between Torch and its
# extension as well as CUDA ABI.
install_requires = ['numpy', f'torch=={torch_version}']

setup(name='fewbit',
      version=version,
      install_requires=install_requires,
      ext_modules=[CMakeExtension('fewbit.fewbit')],
      cmdclass={'build_ext': build_ext})
