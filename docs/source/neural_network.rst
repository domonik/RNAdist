Neural Network
##############

RNAdist features a Neural Network suite that enables training and usage of neural network to predict expected distances
between pairs of nucleotides. It is possible to either use command line calls or a Python API to do so.



Command Line Interface
----------------------

The different calls and parameters  are also explained in the :ref:`DISTAtteNCionE Command Line Interface<datt-cli-doc>`


Generate Training Set
+++++++++++++++++++++

Generating a training set via the command line is easy. You just need a FASTA file containing a sufficient amount of
RNA sequences. Make sure it only contains Letters from `{A, U, C, G}` since others are not supported yet.

.. code-block:: bash

    DISTAtteNCionE generate_data --input Path/to/fasta.fa --ouput Path/to/output/directory --num_threads 10 --nr_samples 1000

The CLI only supports the ability to change a subset of the model details used for ViennaRNA structure sampling.
If you want to train a model for a different setting, consider using the Python API

Training
++++++++

You can train a DISTAttenCioNE neural network via the command line by simply typing


.. code-block:: bash

    DISTAtteNCionE train --input Path/to/input --label_dir Path/to/labels/from/generate_labels --output Path/to/trained_model --data_path /path/to/store/dataset --alpha 0.9 --masking True --learning_rate 0.001 --batch_size 4 --weight_decay 0.005 --device cuda --model normal --nr_layers 1 --gradient_accumulation 1



Prediction
++++++++++

Once you have a trained model you can use it to predict on new, unseen sequences.
This you can achieve via using the following command line call:

.. code-block:: bash

    DISTAtteNCionE predict --input Path/to/input.fa --output Path/to/pckled_output.pckl --model_file /path/to/model --batch_size 4 --device cuda


Python API
----------

Generate Training Set
+++++++++++++++++++++

Training sets are generated via :func:`~RNAdist.nn.training_set_generation.training_set_from_fasta`

Training
++++++++

Training in Python is run via :func:`~RNAdist.nn.training.train_network`

Prediction
++++++++++

If you want to use the python API for prediction have a look at the documentation of the prediction :func:`~RNAdist.nn.prediction.model_predict`
