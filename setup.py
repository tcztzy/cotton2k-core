import logging
import os
from multiprocessing import cpu_count

import numpy
from Cython.Build import cythonize
from pyproject_toml import setup
from setuptools import Extension

log = logging.getLogger("COTTON2K")


def get_extensions():
    extensions = cythonize(
        Extension(
            "cotton2k.core._simulation",
            ["cotton2k/core/_simulation.pyx"],
            include_dirs=[numpy.get_include()],
        ),
        nthreads=cpu_count() if os.name != "nt" else 0,
    )
    return extensions


setup(
    packages=["cotton2k.core"],
    package_data={"cotton2k.core": ["*.json", "*.csv"]},
    ext_modules=get_extensions(),
)
