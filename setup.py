from setuptools import setup, find_packages
import versioneer

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
    install_requires=["torch", "torchvision", "torchaudio", "networkx", "biopython", "smac"],
    setup_requires=["pytest-runner"],
    tests_require=["pytest"],
    scripts=[
        "executables.py"
    ],
    entry_points={
        "console_scripts": [
            "DISTAtteNCionE = executables:main",
        ]
    },
)