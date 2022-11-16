
RNAdist documentation
=====================
.. toctree::
   :maxdepth: 1
   :hidden:

   installation
   tutorial
   api_doc
   cli
   bibliography

--------------------------------------


**Version**: |release|

--------------------------------------


RNAdist is a tool suite to approximate expected distances of nucleotides on the thermodynamic ensemble of possible
RNA secondary structures. It offers a simple Command line interface as well as a PythonAPI to run algorithms such as
"`Boltzmann Sampling`"
and the one proposed in "`Expected distance between terminal nucleotides of RNA secondary structures`"  (:cite:t:`clote2012expected`)
to calculate expected distances.
It therefore mainly uses the C API of the
`ViennaRNA package <https://www.tbi.univie.ac.at/RNA/>`_ (:cite:t:`lorenz2011viennarna`)
Further it is possible to use Neural Networks to predict expected distances given a RNA sequence.

--------------------------------------

.. grid:: 1 1 2 2
    :gutter: 3

    .. grid-item-card::
        :img-top: ../source/_static/DownloadIcon.svg
        :class-card: on-background border rounded border-light
        :class-header: text-center font-weight-bold


        Installation
        ^^^^^^^^^^^^

        Get Started and Install RNAdist

        +++

        .. button-ref:: installation
            :expand:
            :color: secondary
            :click-parent:

            To installation guide

    .. grid-item-card::
        :img-top: ../source/_static/TutorialIcon.svg
        :class-card: on-background border rounded border-light
        :class-header: text-center font-weight-bold



        Tutorial
        ^^^^^^^^^^^^

        Various Tutorials on how to calculate expected distances of
        binding sites and how to train a custom Neural Network to predict
        expected Distances


        +++

        .. button-ref:: tutorial
            :expand:
            :color: secondary
            :click-parent:

            To the tutorial


    .. grid-item-card::
        :img-top: ../source/_static/ConsoleIcon.svg
        :class-card: on-background border rounded border-light
        :class-header: text-center font-weight-bold



        Command Line Interface
        ^^^^^^^^^^^^^^^^^^^^^^

        Automatically generated CLI documentation of RNAdist


        +++

        .. button-ref:: cli
            :expand:
            :color: secondary
            :click-parent:

            To the CLI Documentation

    .. grid-item-card::
        :img-top: ../source/_static/PythonIcon.svg
        :class-card: on-background border rounded border-light
        :class-header: text-center font-weight-bold



        Python API
        ^^^^^^^^^^^^

        Automatically generated API documentation of RNAdist


        +++

        .. button-ref:: api_doc
            :expand:
            :color: secondary
            :click-parent:

            To the API Documentation
