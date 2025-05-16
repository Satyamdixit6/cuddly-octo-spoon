from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="transformer_cuda",
    ext_modules=[
        CUDAExtension(
            "transformer_cuda",
            ["transformer_cuda.cu"],
            extra_compile_args={"cxx": ["-O2"], "nvcc": ["-O2"]}
        )
    ],
    cmdclass={"build_ext": BuildExtension}
)