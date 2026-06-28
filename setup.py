from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="flash_attention_cuda",
    ext_modules=[
        CUDAExtension(
            name="src.attention_cuda",
            sources=[
                "src/attention_cuda.cpp",
                "src/attention_kernel.cu",
            ],
        )
    ],
    cmdclass={
        "build_ext": BuildExtension,
    },
)