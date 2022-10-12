Installation
############


Linux
*****

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

    Most of the required packages will be installed. However, the ViennaRNA_ package needs to be in PATH.
    For this reason we highly recommend to install ViennaRNA and check if the Python interface is installed correctly.
    This can be achieved e.g. via

    .. code-block::

        conda install viennarna=2.5.1 -c bioconda

    if you have issues doing that create a fresh environment and install ViennaRNA before any other dependencies.
    Expecially you need to install it before installing python.

    Further if you receive compiler errors during installation we recommend to use conda g++ and gcc compilers. Both
    should be installed if you type

     .. code-block::

        conda install gxx -c conda-forge


    .. _ViennaRNA: https://www.tbi.univie.ac.at/RNA/

.. note::
    You can test whether everything is installed correctly if you install pytest and run the following command:

    .. code-block::

        pytest --pyargs RNAdist




