"""Build configuration for the PyTorch CUDA extension.

PyTorch must be installed before this file is evaluated. On Kaggle, build with
``--no-build-isolation`` so the extension links against the notebook's existing
PyTorch/CUDA stack instead of downloading a second one.
"""

from __future__ import annotations

import os
from pathlib import Path

from setuptools import setup


ROOT = Path(__file__).resolve().parent


def build_extensions():
    if os.environ.get("FLASH_ATTENTION_SKIP_CUDA_BUILD") == "1":
        return [], {}

    try:
        from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required to build the CUDA extension. Install PyTorch first, "
            "then rerun pip with --no-build-isolation."
        ) from exc

    if CUDA_HOME is None:
        raise RuntimeError(
            "CUDA_HOME could not be detected. Install a CUDA toolkit with nvcc, or set "
            "FLASH_ATTENTION_SKIP_CUDA_BUILD=1 for a reference-only CPU installation."
        )

    extension = CUDAExtension(
        name="flash_attention_cuda._C",
        sources=[
            str(ROOT / "src" / "attention_cuda.cpp"),
            str(ROOT / "src" / "attention_kernel.cu"),
        ],
        extra_compile_args={
            "cxx": ["-O3", "-std=c++17"],
            "nvcc": ["-O3", "--std=c++17", "-lineinfo"],
        },
    )
    command_classes = {
        "build_ext": BuildExtension.with_options(use_ninja=True),
    }
    return [extension], command_classes


EXT_MODULES, CMDCLASS = build_extensions()

setup(
    ext_modules=EXT_MODULES,
    cmdclass=CMDCLASS,
    zip_safe=False,
)
