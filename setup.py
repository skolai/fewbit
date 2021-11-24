from os.path import dirname
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(name='fewbit',
      ext_modules=[
          CppExtension('fewbit',
                       ['fewbit/fewbit.cc', 'fewbit/python_module.cc'],
                       include_dirs=[dirname(__file__)])
      ],
      cmdclass={'build_ext': BuildExtension})
