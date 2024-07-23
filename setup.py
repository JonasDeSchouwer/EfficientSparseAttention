from setuptools import setup, Extension
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(
    name="sparse_attention",
    ext_modules=[
        CppExtension(
            name="sparse_attention",
            sources=["src/sparse_attention.cpp"],
            extra_compile_args=[
                # "-g",  # Enable debugging symbols
                # "-O0",  # Disable any optimizations
                # "-fno-inline",  # Disable function inlining
                "-O3",  # Enable optimizations
                "-fopenmp",  # Enable OpenMP
            ],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
