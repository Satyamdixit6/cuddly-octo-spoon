# setup.py
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension # Use CppExtension, PyTorch handles CUDA details

setup(
    name='transformer_block_cpp',  # Name of the Python package
    ext_modules=[
        CppExtension(
            'transformer_block_cpp', # Must match the name passed to PYBIND11_MODULE
            ['transformer_block.cu'] # Source file (can be .cpp or .cu)
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    })

# Command to build: python setup.py install
# Or for development: python setup.py build_ext --inplace
