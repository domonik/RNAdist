Prediction
##########

Here we will describe how to use trained network to predict
expected distances.

Command Line Interface
----------------------

Once you have a trained model you can use it to predict on new, unseen sequences.
This you can achieve via using the following command line call:

.. code-block:: bash

    DISTAtteNCionE predict --input Path/to/input.fa --output Path/to/pckled_output.pckl --model_file /path/to/model --batch_size 4 --device cuda

Python API
----------

If you want to use the python API for prediction have a look at the documentation of the
:doc:`normal prediction mode <_autosummary/RNAdist.NNModels.prediction.model_predict>`
as well as the
:doc:`window prediction mode <_autosummary/RNAdist.NNModels.prediction.model_window_predict>`


