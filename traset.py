
from torch.utils.cpp_extension import BuildExtension, CppExtension 

setup(
    name='transformer_block_cpp',  
    ext_modules=[
        CppExtension(
            'transformer_block_cpp',
            ['transformer_block.cu'] 
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    })

# Command to build: python setup.py install

