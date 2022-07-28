Visualization
=============

Run It
------

It is possible to visualize all kind of predictions using the shipped Dashboard

Therefore simply type:


.. code-block:: bash

    RNAdist visualize --input path/to/pediction/file.pckl

This will run the dash server using the default port and ip address. You can now use the application if you open your
browser and type http://127.0.0.1:8080/ into the url bar.

RNAdist Visualizer
------------------

Now that you started the visualization. It is time to talk about what you see here.
In the bottom right you can find the selector pane. Here you can select the Fasta header of the sequence you want to
inspect.

.. note::

    The Dropdown Menu is restricted to 100 entries. If your prediction file has more sequences, start typing the name
    of the sequence you search for and it will show up.

Via the Nucleotide Index input you can specify for which nucleotide expected distances should be shown in the upper
distance graph.

Last but not least there is a heatmap in the lower left corner showing distances for all nucleotide combinations

