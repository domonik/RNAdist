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

.. warning::

    Most of the required packages will be installed when using :code:`pip install RNAdist`.
    However, the ViennaRNA_ package needs to be in PATH during install and runtime.
    Since it is not possible to achieve this via a single :code:`pip install` call we highly
    recommend to use anaconda for environment setup and install the package without dependencies.


    .. _ViennaRNA: https://www.tbi.univie.ac.at/RNA/


.. code-block::

	conda install -c conda-forge -c bioconda -c defaults  'viennarna>=2.5 biopython pandas plotly 'dash>=2.5' dash-bootstrap-components pybind11 cython 'python>=3.10' pip versioneer

After this you can install RNAdist via:

.. code-block::

	pip install RNAdist --no-build-isolation --no-deps

.. note::
    You can test whether everything is installed correctly if you **install pytest** and run the following command:

    .. code-block::

        pytest --pyargs RNAdist




