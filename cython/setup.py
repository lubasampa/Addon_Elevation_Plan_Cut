from setuptools import Extension, setup
from Cython.Build import cythonize

extensions = [
    Extension(
        name="meshcut_parallel",
        sources=["meshcut_parallel.pyx"],
        extra_compile_args=["-O3", "-fopenmp"],
        extra_link_args=["-fopenmp"],
    )
]

setup(
    name="meshcut_parallel",
    ext_modules=cythonize(extensions, language_level=3),
)
