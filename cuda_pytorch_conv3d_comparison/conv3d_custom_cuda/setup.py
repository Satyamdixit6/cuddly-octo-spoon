from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# Get the directory of the CUTLASS library.
# You might need to set an environment variable CUTLASS_DIR or adjust this path.
# For simplicity, this example doesn't explicitly link CUTLASS headers/libs,
# as the provided kernel is a naive one. If you were using CUTLASS, you'd add
# include_dirs and library_dirs for CUTLASS here.
# cutlass_dir = os.getenv('CUTLASS_DIR', '/path/to/cutlass') # Example

setup(
    name='conv3d_custom_cuda_op',
    ext_modules=[
        CUDAExtension(
            name='conv3d_custom_cuda_op._C', # The name of the compiled *.so file will reflect this path
            sources=[
                'conv3d_cuda.cpp',
                'conv3d_cuda_kernel.cu',
            ],
            # Example for CUTLASS:
            # include_dirs=[
            #     os.path.join(cutlass_dir, 'include'),
            #     os.path.join(cutlass_dir, 'tools/util/include')
            # ],
            # extra_compile_args={
            #     'cxx': ['-std=c++17', '-O3'],
            #     'nvcc': ['-std=c++17', '-O3', '--use_fast_math', '-Xcompiler', '-Wall,-Wextra']
            # }
            extra_compile_args={
                'cxx': ['-std=c++17', '-O2'], # -O3 can sometimes be too aggressive
                'nvcc': ['-std=c++17', '-O2', # -O3
                         # Example: For a specific architecture, e.g., Ampere
                         # '-gencode=arch=compute_80,code=sm_80',
                         # Or use a common one like Volta if you don't know the target arch
                         # '-gencode=arch=compute_70,code=sm_70',
                         # Or even more general, but might not be optimal:
                         # '-gencode=arch=compute_52,code=sm_52', # Maxwell
                         # '-gencode=arch=compute_61,code=sm_61', # Pascal
                         # '-gencode=arch=compute_75,code=sm_75', # Turing
                        ]
            }
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })