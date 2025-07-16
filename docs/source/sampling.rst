.. _sampling-ref:

Sampling
########

You can use Boltzmann sampling to approximate expected distances using either the Command Line Interface or the
Python API

Python API
----------

.. currentmodule:: RNAdist.sampling.ed_sampling


The Python API is fully compatible with the ViennaRNA Python fold compound. Thus, you can set up a fold compound and use
the :func:`~sample_fc` function to get a numpy array containing all pairwise distances. Have
look into the function documentation for details and examples. If you are only interested in the distance between two
nucleotides :code:`i` and :code:`j`, you can speed up this procedure by using :func:`~sample_distance_ij`.


Command Line Interface
----------------------

For approximation via the CLI you need a FASTA file with your sequences in it. This is passed via the :code:`--input`
flag. The File specified in :code:`--output` will contain a pickled python dictionary with the FASTA identifiers as keys
and numpy arrays containing all pairwise expected distances as values.

.. code-block:: bash

	RNAdist sample --input FASTA --output OUTPUT --num_threads 2 --nr_samples 1000

In contrast to the Python API it does not support
constraint folding. To see the documentation of how to visualize the output and further CLI flagst have a look into the
CLI documentation :ref:`RNAdist Command Line Interface<cli-doc>`.