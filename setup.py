from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='my_custom_ops',
    ext_modules=[
        CUDAExtension('my_custom_ops_cuda', [
            'my_ops.cpp',
            'my_kernels.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })