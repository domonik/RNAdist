Training
########

Here we will describe how to train the neural network

Command Line Interface
----------------------

You can train a DISTAttenCioNE neural network via the command line by simply typing


.. code-block:: bash

    DISTAtteNCionE train --input Path/to/input --label_dir Path/to/labels/from/generate_labels --output Path/to/trained_model --data_path /path/to/store/dataset --alpha 0.9 --masking True --learning_rate 0.001 --batch_size 4 --weight_decay 0.005 --device cuda --model normal --nr_layers 1 --gradient_accumulation 1

Python API
----------

The documentation for the python API can be found :doc:`here <_autosummary/RNAdist.NNModels.training.train_network>`

