Installation
############

.. note::
    Best practice is to install RNAdist into an encapsulated environment e.g. via Conda:

    .. code-block::

        conda create -n rnadist_env
        conda activate rnadist_env


Conda
-----

Building a conda package is in planning but not done yet

Pip
---

The Pip Package can be installed easily via:

.. code-block::

    pip install rnadist

.. warning::

    Most of the required packages will be installed. However, the ViennaRNA_ package needs to be in PATH. Also, make sure that the Python API of the
    ViennaRNA package is installed. Keep in mind that for some functionality you need a special version of this package that allows access to
    DP matrices via the python API.

    .. _ViennaRNA: https://www.tbi.univie.ac.at/RNA/


