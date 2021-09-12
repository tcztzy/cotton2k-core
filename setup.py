import logging
import os
import subprocess
from collections import defaultdict
from glob import glob
from multiprocessing import cpu_count
from pathlib import Path

import numpy
import setuptools
from Cython.Build import build_ext, cythonize
from pyproject_toml import setup
from setuptools.command.develop import develop
from setuptools.extension import Extension

log = logging.getLogger("COTTON2K")


extra_compile_args = defaultdict(lambda: ["-std=c++20"])
extra_compile_args["msvc"] = ["/std:c++latest"]
libraries = defaultdict(lambda: ["cotton2k"])
libraries["nt"].extend(["ws2_32", "userenv", "advapi32"])


class cotton2k_build_ext(build_ext):
    def build_extensions(self):
        cargo_build = ["cargo", "build"]
        if not self.debug:
            cargo_build.append("--release")
        subprocess.run(cargo_build)
        args = extra_compile_args[self.compiler.compiler_type]
        for ext in self.extensions:
            ext.library_dirs = [
                os.path.join("target", "debug" if self.debug else "release")
            ]
            ext.extra_compile_args = args
        super().build_extensions()


class cotton2k_develop(develop):
    def install_for_development(self):
        # Without 2to3 inplace works fine:
        self.run_command("egg_info")

        # Build extensions in-place
        self.reinitialize_command("build_ext", inplace=1, debug=1)
        self.run_command("build_ext")

        if setuptools.bootstrap_install_from:
            self.easy_install(setuptools.bootstrap_install_from)
            setuptools.bootstrap_install_from = None

        self.install_namespaces()

        # create an .egg-link in the installation dir, pointing to our egg
        log.info("Creating %s (link to %s)", self.egg_link, self.egg_base)
        if not self.dry_run:
            with open(self.egg_link, "w") as f:
                f.write(self.egg_path + "\n" + self.setup_path)
        # postprocess the installed distro, fixing up .pth, installing scripts,
        # and handling requirements
        self.process_distribution(None, self.dist, not self.no_deps)


def get_extensions():
    extensions = cythonize(
        "src/_cotton2k/*.pyx",
        nthreads=cpu_count() if os.name != "nt" else 0,
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
    cmdclass={"build_ext": cotton2k_build_ext, "develop": cotton2k_develop},
)
