Binding Site Distance
#####################

Just like discussed in the corresponding paper, we showed that the clote-ponty algorithm is capable of
calculating expected distances of exterior nucleotides. Since interaction of mRNA with molecules such as RNA binding
proteins or regulatory RNAs prevents the binding-site from forming intramolecular hydrogen bonds, the binding-sites
are forced to be exterior.

Here we will shortly describe how to calulate the expected distance of two binding-sites using the RNAdist Python API.
Therefore we picked binding-sites of PUM2 a famous RNA binding protein.

.. note::

    The binding sites were extracted from an publically available e-CLIP dataset and processed to find the most likely
    transcript variant they originated from using the tool `Peakhood <https://github.com/BackofenLab/Peakhood>`_.
    Also we shrank the binding site to mainly contain a learned interaction motif of PUM2 using
    `RNAProt <https://github.com/BackofenLab/RNAProt>`_.

    **BindingSites** ::

        ENST00000261313.3	715 721
        ENST00000261313.3	931 937

First we need to set up a ViennaRNA fold compound and add the constraints at the binding sites. This will force the
sites to be exterior on the ensemble of RNA graphs.

    >>> import RNA
    >>> seq = "AGUGUGCUGAGCUCUCCGCGUCGCCUCUGUCGCCCGCGCCUGGCCUACCGCGGCACUCCCGGCUGCACGCUCUGCUUGGCCUCGCCAUGCCGGUGGACCUCAGCAAGUGGUCCGGGCCCUUGAGCCUGCAAGAAGUGGACGAGCAGCCGCAGCACCCGCUGCAUGUCACCUACGCCGGGGCGGCGGUGGACGAGCUGGGCAAAGUGCUGACGCCCACCCAGGUUAAGAAUAGACCCACCAGCAUUUCGUGGGAUGGUCUUGAUUCAGGGAAGCUCUACACCUUGGUCCUGACAGACCCGGAUGCUCCCAGCAGGAAGGAUCCCAAAUACAGAGAAUGGCAUCAUUUCCUGGUGGUCAACAUGAAGGGCAAUGACAUCAGCAGUGGCACAGUCCUCUCCGAUUAUGUGGGCUCGGGGCCUCCCAAGGGCACAGGCCUCCACCGCUAUGUCUGGCUGGUUUACGAGCAGGACAGGCCGCUAAAGUGUGACGAGCCCAUCCUCAGCAACCGAUCUGGAGACCACCGUGGCAAAUUCAAGGUGGCGUCCUUCCGUAAAAAGUAUGAGCUCAGGGCCCCGGUGGCUGGCACGUGUUACCAGGCCGAGUGGGAUGACUAUGUGCCCAAACUGUACGAGCAGCUGUCUGGGAAGUAGGGGGUUAGCUUGGGGACCUGAACUGUCCUGGAGGCCCCAAGCCAUGUUCCCCAGUUCAGUGUUGCAUGUAUAAUAGAUUUCUCCUCUUCCUGCCCCCCUUGGCAUGGGUGAGACCUGACCAGUCAGAUGGUAGUUGAGGGUGACUUUUCCUGCUGCCUGGCCUUUAUAAUUUUACUCACUCACUCUGAUUUAUGUUUUGAUCAAAUUUGAACUUCAUUUUGGGGGGUAUUUUGGUACUGUGAUGGGGUCAUCAAAUUAUUAAUCUGAAAAUAGCAACCCAGAAUGUAAAAAAGAAAAAACUGGGGGGAAAAAGACCAGGUCUACAGUGAUAGAGCAAAGCAUCAAAGAAUCUUUAAGGGAGGUUUAAAAAAAAAAAAAAAAAAAAAGAUUGGUUGCCUCUGCCUUUGUGAUCCUGAGUCCAGAAUGGUACACAAUGUGAUUUUAUGGUGAUGUCACUCACCUAGACAACCAGAGGCUGGCAUUGAGGCUAACCUCCAACACAGUGCAUCUCAGAUGCCUCAGUAGGCAUCAGUAUGUCACUCUGGUCCCUUUAAAGAGCAAUCCUGGAAGAAGCAGGAGGGAGGGUGGCUUUGCUGUUGUUGGGACAUGGCAAUCUAGACCGGUAGCAGCGCUCGCUGACAGCUUGGGAGGAAACCUGAGAUCUGUGUUUUUUAAAUUGAUCGUUCUUCAUGGGGGUAAGAAAAGCUGGUCUGGAGUUGCUGAAUGUUGCAUUAAUUGUGCUGUUUGCUUGUAGUUGAAUAAAAAUAGAAACCUGAAUGAAGGAA"
    >>> fc = RNA.fold_compound(seq)
    >>> for x in range(931, 937):
    ...     fc.hc_add_up(x)
    >>> for x in range(715, 721):
    ...     fc.hc_add_up(x)


Now that we set up the ViennaRNA fold compound object it is time to calculate the partition function and use the Python
API of RNAdist to get the expected distance matrix

.. Note::

    At this point your ViennaRNA Python API needs to have access to the dynamic programming matrices.
    This should be the case if you use the latest RNAdist version.

.. code-block:: python

    >>> from RNAdist.DPModels.clote import compute_clote
    >>> _ = fc.pf()
    >>> expected_distance = compute_clote(fc)


We finally got the expected Distance for each nucleotide to each other nucleotide. Now the interesting distance
if obviously the distance from nucleotide :code:`720` to  :code:`937`. This you will get by just indexing the array


.. code-block:: python

    >>> interesting_distance = expected_distance[720][937]
    >>> interesting_distance
    34.87043018944288
    >>> backbone_distance = 937 - 720
    217

As you can see the expected distance between the two binding sites is much different from the backbone distance.