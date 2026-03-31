import sys
from pathlib import Path

from Cython.Build import cythonize
from setuptools import Extension, setup


PACKAGE_NAME = "native_backend"
PACKAGE_DIR = Path(__file__).resolve().parent.parent / PACKAGE_NAME


if sys.platform.startswith("win"):
    compile_args = ["/O2", "/openmp"]
    link_args = []
else:
    compile_args = ["-O3", "-fopenmp"]
    link_args = ["-fopenmp"]


extensions = [
    Extension(
        name=f"{PACKAGE_NAME}.meshcut_parallel",
        sources=["meshcut_parallel.pyx"],
        extra_compile_args=compile_args,
        extra_link_args=link_args,
    )
]


setup(
    name="meshcut_parallel",
    packages=[PACKAGE_NAME],
    package_dir={PACKAGE_NAME: str(PACKAGE_DIR)},
    ext_modules=cythonize(extensions, language_level=3),
)
