from setuptools import setup, find_packages
import versioneer
from Cython.Build import cythonize
import numpy as np

NAME = "RNAdist"
DESCRIPTION = "Package for Calculating Expected Distances on the " \
              "ensemble of RNA structures"

# Set __version__ done by versioneer
# exec(open("NextSnakes/__init__.py").read())

setup(
    name=NAME,
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    author="domonik",
    author_email="dominik.rabsch@gmail.com",
    packages=find_packages(),
    license="LICENSE",
    url="https://github.com/domonik/RNAdist",
    description=DESCRIPTION,
    long_description=DESCRIPTION,
    long_description_content_type="text/markdown",
    include_package_data=True,
    package_data={"RNAdist.visualize": ["assets/*"]},
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
    ext_modules=cythonize("RNAdist/DPModels/_dp_calulations.pyx"),
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