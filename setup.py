from setuptools import setup, find_packages, Extension
import versioneer
from Cython.Build import cythonize
from torch.utils import cpp_extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
import numpy as np
import os
import sys
import pybind11

NAME = "RNAdist"
DESCRIPTION = "Package for Calculating Expected Distances on the " \
              "ensemble of RNA structures"

with open("README.md") as handle:
    LONGDESC = handle.read()


extra_link_args = []
include_dir = []
prefix = os.getenv("CONDA_PREFIX")
if prefix is None:
    print("No conda environment found. Falling back to default installation path of ViennaRNA.\n"
          "Consider using a conda environment instead")
    prefix = "/usr/local"

_include = os.path.join(prefix, "include")
_lib = os.path.join(prefix, "lib")
extra_link_args += [f"-L{_lib}", f"-I{_include}"]
include_dir += [_include] + [np.get_include()]
svi = sys.version_info
python_l = f"python{svi[0]}.{svi[1]}"

if not os.path.exists(os.path.join(prefix, "lib", "libRNA.a")):
    raise FileNotFoundError("Not able to find ViennaRNA RNAlib installation. This version of RNAdist requires ViennaRNA "
                            "to be installed. You can easily install it using Conda:\n"
                            "conda install -c bioconda viennarna"
                            )
if not os.path.exists(os.path.join(prefix, "lib", python_l, "site-packages", "RNA")):
    raise ImportError("Not able to find ViennaRNA python package in your current environment."
                      "Please install it e.g. via Conda\n"
                        "conda install -c bioconda viennarna"
)

sampling_extension = Pybind11Extension(
    "RNAdist.sampling.cpp.sampling",
    sources=[
        "RNAdist/sampling/cpp/edsampling.cpp",
        "RNAdist/sampling/cpp/RNAGraph.cpp",
        "RNAdist/sampling/cpp/pyedsampling.cpp"
    ],
    extra_link_args=[f"-I{pybind11.get_include()}"] + extra_link_args + ["-lRNA", "-lpthread", "-lstdc++", "-fopenmp", "-lm", f"-l{python_l}",
                                                                         "-Wl,--no-undefined"],
    include_dirs=[_include, pybind11.get_include()],
    language="c++"
)

cp_exp_dist_extension = Extension(
    "CPExpectedDistance.c_expected_distance",
    sources=["CPExpectedDistance/CPExpectedDistance/c_expected_distance.c"],
    extra_link_args=extra_link_args + ["-lRNA", "-lpthread", "-lstdc++", "-fopenmp", "-lm", f"-l{python_l}", "-Wl,--no-undefined"],
    include_dirs=include_dir,
    language="c"
)


class CustomBuildExtension(cpp_extension.BuildExtension):

    def __init__(self, *args, **kwargs):
        # This has to stay until I rewrite the Clote-Ponty Extension or find out how to force ninja not to use
        # a c++ compiler for that extension
        kwargs["use_ninja"] = False
        super().__init__(*args, **kwargs)



cmds = versioneer.get_cmdclass()
cmds["build_ext"] = CustomBuildExtension
setup(
    name=NAME,
    version=versioneer.get_version(),
    cmdclass=cmds,
    author="domonik",
    author_email="dominik.rabsch@gmail.com",
    packages=find_packages() + find_packages("CPExpectedDistance"),
    package_dir={"RNAdist": "./RNAdist", "CPExpectedDistance": "CPExpectedDistance/CPExpectedDistance"},
    license="LICENSE",
    url="https://github.com/domonik/RNAdist",
    description=DESCRIPTION,
    long_description=LONGDESC,
    long_description_content_type="text/markdown",
    include_package_data=True,
    package_data={
        "RNAdist.visualize": ["assets/*"],
        "RNAdist": ["tests/*.py", "tests/test_data/*"],
        "RNAdist.dp": ["tests/*.py", "tests/test_data/*"],
        "RNAdist.nn": ["tests/*.py", "tests/test_data/*.fa", "tests/test_data/*.pt", "tests/test_data/expected*/*"],
        "RNAdist.sampling": ["tests/*.py", "tests/test_data"],
    },
    install_requires=[
        "torch",
        "torchvision",
        "torchaudio",
        "networkx",
        "biopython",
        "pandas",
        "smac>=1.4",
        "plotly",
        "dash>=2.5",
        "dash_bootstrap_components",
    ],
    setup_requires=["pytest-runner"],
    tests_require=["pytest"],
    ext_modules=[
        cpp_extension.CppExtension(
            name="RNAdist.nn.nn_helpers",
            sources=["RNAdist/nn/nn_helpers.cpp"],
            include_dirs=cpp_extension.include_paths(),
            extra_link_args=["-Wl,--no-undefined", f"-l{python_l}"],
            language="c++"
        ),
    ] + cythonize("RNAdist/dp/_dp_calculations.pyx") + [sampling_extension, cp_exp_dist_extension],
    include_dirs=np.get_include(),
    scripts=[
        "RNAdist/executables.py",
        "versioneer.py"
    ],
    entry_points={
        "console_scripts": [
            "DISTAtteNCionE = RNAdist.distattencione_executables:main",
            "RNAdist = RNAdist.executables:main"
        ]
    },
)