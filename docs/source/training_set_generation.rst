
Generate Training Set
#####################

Here we will describe how to generate a Training set for the
DISTAttenCionE Network

Command Line Interface
----------------------

Generating a training set via the command line is easy. You just need a FASTA file containing a sufficient amount of
RNA sequences. Make sure it only contains Letters from `{A, U, C, G}` since others are not supported yet.

.. code-block:: bash

    DISTAtteNCionE generate_data --input Path/to/fasta.fa --ouput Path/to/output/directory --num_threads 10 --nr_samples 1000

The CLI only supports the ability to change a subset of the model details used for ViennaRNA structure sampling.
If you want to train a model for a different setting, consider using the Python API

Python API
----------

The documentation for the python API can be found :doc:`here <_autosummary/RNAdist.nn.training_set_generation.training_set_from_fasta>`

