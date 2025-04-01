Output and Extraction
#####################

Output files of any of the prediction methods (sampling, clote-ponty, pmcomp) contain just a Python
dictionary of sequence identifiers as keys and numpy arrays as values. It is possible to load it via pythons
pickle module.
You can access the expected distance
between nucleotide :code:`i` and :code:`j` via their corresponding indices in this array. For example the distance
between the first nucleotide and the 20th can be accessed like this:

.. code-block:: python

	import pickle
	with open(output_file, "rb") as handle:
		rnadist_sample_output: Dict[str, np.ndarray] = pickle.load(handle)
	ed1_20 = rnadist_sample_output[0][19]

.. warning::

	In contrast to ViennRNA this Tool uses 0 based indexing

However, if you want to use another
programming Language for further work, the pickled expected distance files can be extracted to TSV files via the command
line. This is done via the following call:

.. code-block:: bash

	RNAdist extract --data_file path/to/predict/output.pckl --outdir example_dir

This will produce a single tsv file for each sequence that was used in the prediction step. All tsv files will
be saved in the outdir and their names will be equal to the FASTA sequence identifiers.
For further understanding have a look at the :ref:`RNAdist Command Line Interface<cli-doc>`.