# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#!/usr/bin/env python

import glob
import os
import platform

import torch
from setuptools import find_packages
from setuptools import setup
from torch.utils.cpp_extension import CUDA_HOME
from torch.utils.cpp_extension import CppExtension
from torch.utils.cpp_extension import CUDAExtension

requirements = ["torch", "torchvision"]


#
# if torch.cuda.is_available():
#     print('Including CUDA code.')
#     sources += ['src/psroi_pooling_cuda.c']
#     headers += ['src/psroi_pooling_cuda.h']
#     defines += [('WITH_CUDA', None)]
#     with_cuda = True
#
# this_file = os.path.dirname(os.path.realpath(__file__))
# print(this_file)
# extra_objects = ['src/cuda/psroi_pooling.cu.o']
# extra_objects = [os.path.join(this_file, fname) for fname in extra_objects]
#
# ffi = create_extension(
#     '_ext.psroi_pooling',
#     headers=headers,
#     sources=sources,
#     define_macros=defines,
#     relative_to=__file__,
#     with_cuda=with_cuda,
#     extra_objects=extra_objects
# )
#
# if __name__ == '__main__':
#     ffi.build()


def get_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, "model", "csrc")

    main_file = glob.glob(os.path.join(extensions_dir, "*.cpp"))
    source_cuda = glob.glob(os.path.join(extensions_dir, "cuda", "*.cu"))

    sources = main_file
    extension = CppExtension
    cxx_flags = []
    if platform.system() == 'Darwin':
        cxx_flags = ["-g", "-stdlib=libc++", "-std=c++11", "-mmacosx-version-min=10.9"]
        platform.release()

    extra_compile_args = {"cxx": cxx_flags}
    define_macros = []

    if torch.cuda.is_available() and CUDA_HOME is not None:
        extension = CUDAExtension
        sources += source_cuda
        define_macros += [("WITH_CUDA", None)]
        extra_compile_args["nvcc"] = [
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ]

    sources = [os.path.join(extensions_dir, s) for s in sources]

    include_dirs = [extensions_dir]

    ext_modules = [
        extension(
            "model._C",
            sources=sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]

    return ext_modules


setup(
    name="psroialign",
    version="1.0.0",
    description="psroialign with pytorch 1.x",
    author="Do Lin",
    author_email="mcdooooo@gmail.com",
    license="MIT",
    packages=find_packages(exclude=("configs", "tests",)),
    # install_requires=requirements,
    ext_modules=get_extensions(),
    cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
)
