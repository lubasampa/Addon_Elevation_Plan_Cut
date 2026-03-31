import sys

from Cython.Build import cythonize
from setuptools import Extension, setup


if sys.platform.startswith("win"):
    compile_args = ["/O2", "/openmp"]
    link_args = []
else:
    compile_args = ["-O3", "-fopenmp"]
    link_args = ["-fopenmp"]


extensions = [
    Extension(
        name="meshcut_parallel",
        sources=["meshcut_parallel.pyx"],
        extra_compile_args=compile_args,
        extra_link_args=link_args,
    )
]


setup(
    name="meshcut_parallel",
    ext_modules=cythonize(extensions, language_level=3),
)
