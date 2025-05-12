from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name="transformer_cpp",
    ext_modules=[
        CUDAExtension(
            "transformer_cpp",
            [
                "attention.cu",
                "moe.cu",
                "transformer.cpp"
            ],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)