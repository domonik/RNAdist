Binding Site Distance
#####################





Binding site distance can be calculated using two different approaches. The first one (Boltzmann sampling) is the suggested use if you dont
have prior knowledge about the kind of strand binding behavior of your molecule. The second one is the recommended use
if you know that your molecules tend to bind exterior regions on the RNA structure. It is recommended to use the
clote-ponty algorithm then, since it actually calculates the expected distance instead of an approximation

Without Prior Information
--------------------------

.. currentmodule:: RNAdist.sampling.ed_sampling

You can approximate the expected distance between two binding sites via sampling.
The fastest way to get the distance between two binding sites in such a case is to use :func:`~sample_distance_ij`.
Please have a look into the :doc:`sampling chapter <sampling>` of the Tutorial if you want to use a CLI for sampling.





Exterior Strand Binders
-----------------------

.. currentmodule:: RNAdist.dp.cpedistance


We showed that the clote-ponty algorithm is capable of
calculating expected distances of exterior nucleotides. Since interaction of mRNA with molecules such as RNA binding
proteins or regulatory RNAs prevents the binding-site from forming intramolecular hydrogen bonds, the binding-sites
might be forced to be in unstructured regions hence  exterior nucleotides. In such cases the mentioned
algorithm will have no error since no structures span the interval between two binding sites. in the following
paragraphs we will show how to calculate expected distances between such binding sites using either the Python API
or the Command Line Interface


Python API
***********

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
    135.0

You can see the mfe distance on this RNA is much different from the expected distance.


Command Line Interface
**********************

The command line interface uses `FASTA` as well as `BED` files to calculate distances between binding sites.
We will consider the same example as the one used in the Tutorial for the Python API. Thus you must have a BED file
which looks the following

**BED** ::

	ENST00000261313.3	715 721
	ENST00000261313.3	931 937


As well as a FASTA file that has got sequences sharing its description with entries in the first column of the bed file.

**FASTA** ::

	>ENST00000261313.3
	AGUGUGCUGAGCUCUCCGCGUCGCCUCUGUCGCCCGCGCCUGGCCUACCGCGGCACUCCCGGCUGCACGCUCUGCUUGGCCUCGCCAUGCCGGUGGACCUCAGCAAGUGGUCCGGGCCCUUGAGCCUGCAAGAAGUGGACGAGCAGCCGCAGCACCCGCUGCAUGUCACCUACGCCGGGGCGGCGGUGGACGAGCUGGGCAAAGUGCUGACGCCCACCCAGGUUAAGAAUAGACCCACCAGCAUUUCGUGGGAUGGUCUUGAUUCAGGGAAGCUCUACACCUUGGUCCUGACAGACCCGGAUGCUCCCAGCAGGAAGGAUCCCAAAUACAGAGAAUGGCAUCAUUUCCUGGUGGUCAACAUGAAGGGCAAUGACAUCAGCAGUGGCACAGUCCUCUCCGAUUAUGUGGGCUCGGGGCCUCCCAAGGGCACAGGCCUCCACCGCUAUGUCUGGCUGGUUUACGAGCAGGACAGGCCGCUAAAGUGUGACGAGCCCAUCCUCAGCAACCGAUCUGGAGACCACCGUGGCAAAUUCAAGGUGGCGUCCUUCCGUAAAAAGUAUGAGCUCAGGGCCCCGGUGGCUGGCACGUGUUACCAGGCCGAGUGGGAUGACUAUGUGCCCAAACUGUACGAGCAGCUGUCUGGGAAGUAGGGGGUUAGCUUGGGGACCUGAACUGUCCUGGAGGCCCCAAGCCAUGUUCCCCAGUUCAGUGUUGCAUGUAUAAUAGAUUUCUCCUCUUCCUGCCCCCCUUGGCAUGGGUGAGACCUGACCAGUCAGAUGGUAGUUGAGGGUGACUUUUCCUGCUGCCUGGCCUUUAUAAUUUUACUCACUCACUCUGAUUUAUGUUUUGAUCAAAUUUGAACUUCAUUUUGGGGGGUAUUUUGGUACUGUGAUGGGGUCAUCAAAUUAUUAAUCUGAAAAUAGCAACCCAGAAUGUAAAAAAGAAAAAACUGGGGGGAAAAAGACCAGGUCUACAGUGAUAGAGCAAAGCAUCAAAGAAUCUUUAAGGGAGGUUUAAAAAAAAAAAAAAAAAAAAAGAUUGGUUGCCUCUGCCUUUGUGAUCCUGAGUCCAGAAUGGUACACAAUGUGAUUUUAUGGUGAUGUCACUCACCUAGACAACCAGAGGCUGGCAUUGAGGCUAACCUCCAACACAGUGCAUCUCAGAUGCCUCAGUAGGCAUCAGUAUGUCACUCUGGUCCCUUUAAAGAGCAAUCCUGGAAGAAGCAGGAGGGAGGGUGGCUUUGCUGUUGUUGGGACAUGGCAAUCUAGACCGGUAGCAGCGCUCGCUGACAGCUUGGGAGGAAACCUGAGAUCUGUGUUUUUUAAAUUGAUCGUUCUUCAUGGGGGUAAGAAAAGCUGGUCUGGAGUUGCUGAAUGUUGCAUUAAUUGUGCUGUUUGCUUGUAGUUGAAUAAAAAUAGAAACCUGAAUGAAGGAA

.. note::

	It is possible to have multiple different sequence identifiers in the BED and FASTA files. However, the FASTA file must
	contain all the entries in the first column of the bed file.

The following command line call will produce a TSV with the expected distance between those binding sites

.. code-block:: bash

	RNAdist binding-site --input my.fasta --bed_files my.bed --ouput my_output.tsv --num_threads 5


.. note::

	You can use as many bed files as you want by separating them using whitespaces.

.. warning::

	RNAdist will automatically filter overlapping sites since the expected distance will be zero


Output
++++++

The output TSV will have the following structure

**TSV** ::

	sequence_name	name1	start1	stop1	name2	start2	stop2	expected_distance
	ENST00000261313.3	foo.bed	715	721	foo.bed	931	937	26.263341302657103

Where :code:`sequence_name` is the fasta header as well as the bed first column. The following columns originate from the
BED files and are the names of the files (or custom names specified via :code:`--names` flag) as well as the intervals.
