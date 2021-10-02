import logging
import os
from collections import defaultdict
from glob import glob
from multiprocessing import cpu_count

import numpy
from Cython.Build import build_ext, cythonize
from pyproject_toml import setup

log = logging.getLogger("COTTON2K")


extra_compile_args = defaultdict(lambda: ["-std=c++20"])
extra_compile_args["msvc"] = ["/std:c++latest"]
libraries = defaultdict(lambda: [])
libraries["nt"].extend(["ws2_32", "userenv", "advapi32"])


class cotton2k_build_ext(build_ext):
    def build_extensions(self):
        args = extra_compile_args[self.compiler.compiler_type]
        for ext in self.extensions:
            ext.extra_compile_args = args
        super().build_extensions()


def get_extensions():
    extensions = cythonize(
        "src/_cotton2k/*.pyx",
        nthreads=cpu_count() if os.name != "nt" else 0,
        language="c++",
        compiler_directives={"language_level": 3},
    )
    for ext in extensions:
        ext.include_dirs = [numpy.get_include()]
        ext.libraries = libraries[os.name]
        ext.sources = glob("src/_cotton2k/*.cpp")
    return extensions


setup(
    packages=["cotton2k.core", "_cotton2k"],
    package_dir={"": "src"},
    package_data={"cotton2k.core": ["*.json", "*.csv"]},
    ext_modules=get_extensions(),
    cmdclass={"build_ext": cotton2k_build_ext},
)
