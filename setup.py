from setuptools import setup, Extension
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(
    name="sparse_attention",
    ext_modules=[
        CppExtension(
            name="sparse_attention",
            sources=["src/sparse_attention.cpp"],
            extra_compile_args=[
                "-g",
                "-O0",
                "-fno-inline",
                # "-O3",
            ],  # Enable debugging symbols and disable any optimizations
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
