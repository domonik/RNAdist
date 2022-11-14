Binding Site Distance
#####################

.. currentmodule:: RNAdist.dp.cpedistance


Just like discussed in the corresponding paper, we showed that the clote-ponty algorithm is capable of
calculating expected distances of exterior nucleotides. Since interaction of mRNA with molecules such as RNA binding
proteins or regulatory RNAs prevents the binding-site from forming intramolecular hydrogen bonds, the binding-sites
are forced to be in unstructured regions such as in a multi loop or as exterior nucleotides. In both cases the mentioned
algorithm will have no error since no basepairs span into the interval between two binding sites.

Here we will shortly describe how to calulate the expected distance of two binding-sites using the RNAdist Python API.
Therefore we picked binding-sites of PUM2 a famous RNA binding protein.

.. note::

    The binding sites were extracted from an publically available e-CLIP dataset and processed to find the most likely
    transcript variant. They originated from using the tool `Peakhood <https://github.com/BackofenLab/Peakhood>`_.
    Also we shrank the binding site to mainly contain a learned interaction motif of PUM2 using
    `RNAProt <https://github.com/BackofenLab/RNAProt>`_.

    **BindingSites** ::

        ENST00000261313.3	715 721
        ENST00000261313.3	931 937

First we need to set up a ViennaRNA fold compound and generate a list with those to binding sites.

    >>> import RNA
    >>> seq = "AGUGUGCUGAGCUCUCCGCGUCGCCUCUGUCGCCCGCGCCUGGCCUACCGCGGCACUCCCGGCUGCACGCUCUGCUUGGCCUCGCCAUGCCGGUGGACCUCAGCAAGUGGUCCGGGCCCUUGAGCCUGCAAGAAGUGGACGAGCAGCCGCAGCACCCGCUGCAUGUCACCUACGCCGGGGCGGCGGUGGACGAGCUGGGCAAAGUGCUGACGCCCACCCAGGUUAAGAAUAGACCCACCAGCAUUUCGUGGGAUGGUCUUGAUUCAGGGAAGCUCUACACCUUGGUCCUGACAGACCCGGAUGCUCCCAGCAGGAAGGAUCCCAAAUACAGAGAAUGGCAUCAUUUCCUGGUGGUCAACAUGAAGGGCAAUGACAUCAGCAGUGGCACAGUCCUCUCCGAUUAUGUGGGCUCGGGGCCUCCCAAGGGCACAGGCCUCCACCGCUAUGUCUGGCUGGUUUACGAGCAGGACAGGCCGCUAAAGUGUGACGAGCCCAUCCUCAGCAACCGAUCUGGAGACCACCGUGGCAAAUUCAAGGUGGCGUCCUUCCGUAAAAAGUAUGAGCUCAGGGCCCCGGUGGCUGGCACGUGUUACCAGGCCGAGUGGGAUGACUAUGUGCCCAAACUGUACGAGCAGCUGUCUGGGAAGUAGGGGGUUAGCUUGGGGACCUGAACUGUCCUGGAGGCCCCAAGCCAUGUUCCCCAGUUCAGUGUUGCAUGUAUAAUAGAUUUCUCCUCUUCCUGCCCCCCUUGGCAUGGGUGAGACCUGACCAGUCAGAUGGUAGUUGAGGGUGACUUUUCCUGCUGCCUGGCCUUUAUAAUUUUACUCACUCACUCUGAUUUAUGUUUUGAUCAAAUUUGAACUUCAUUUUGGGGGGUAUUUUGGUACUGUGAUGGGGUCAUCAAAUUAUUAAUCUGAAAAUAGCAACCCAGAAUGUAAAAAAGAAAAAACUGGGGGGAAAAAGACCAGGUCUACAGUGAUAGAGCAAAGCAUCAAAGAAUCUUUAAGGGAGGUUUAAAAAAAAAAAAAAAAAAAAAGAUUGGUUGCCUCUGCCUUUGUGAUCCUGAGUCCAGAAUGGUACACAAUGUGAUUUUAUGGUGAUGUCACUCACCUAGACAACCAGAGGCUGGCAUUGAGGCUAACCUCCAACACAGUGCAUCUCAGAUGCCUCAGUAGGCAUCAGUAUGUCACUCUGGUCCCUUUAAAGAGCAAUCCUGGAAGAAGCAGGAGGGAGGGUGGCUUUGCUGUUGUUGGGACAUGGCAAUCUAGACCGGUAGCAGCGCUCGCUGACAGCUUGGGAGGAAACCUGAGAUCUGUGUUUUUUAAAUUGAUCGUUCUUCAUGGGGGUAAGAAAAGCUGGUCUGGAGUUGCUGAAUGUUGCAUUAAUUGUGCUGUUUGCUUGUAGUUGAAUAAAAAUAGAAACCUGAAUGAAGGAA"
    >>> fc = RNA.fold_compound(seq)
    >>> binding_sites = [(715, 721), (931, 937)]



Now that we set up the ViennaRNA fold compound object and the binding sites list we can use the helper function
:func:`binding_site_distance`


.. code-block:: python

    >>> from RNAdist.dp.cpedistance import binding_site_distance
    >>> interesting_distance = binding_site_distance(fc, binding_sites)


We finally got the expected distance between nucleotides :code:`720` and  :code:`937`.



.. code-block:: python

    >>> interesting_distance
    26.263341302657103
    >>> backbone_distance = 931 - 720
    >>> backbone_distance
    211

As you can see the expected distance between the two binding sites is much different from the backbone distance.

You might also wonder how different it is from the MFE  structure. Therefore we use the ViennRNA API again to get
the string representation of the MFE. Further we import a helper function that can calculate pairwise distances from
a secondary structure.

.. code-block:: python

    >>> from RNAdist.sampling.ed_sampling import shortest_paths_from_struct
    >>> mfe_structure, _ = fc.mfe()
    >>> ed_matrix = shortest_paths_from_struct(mfe_structure)


Now we just use the binding site indices to get the mfe distance

.. code-block:: python

    >>> ed_matrix[720][931]
    29

This time the expected distance is only a bit different from the MFE distance. However, depending on the RNA and
binding sites this might change.