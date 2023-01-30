import RNA
import numpy as np
from tempfile import NamedTemporaryFile
import subprocess
import os
import networkx as nx
from typing import List, Dict, Iterable
from RNAdist.sampling.cpp.sampling import cpp_nr_sampling, cpp_sampling, cpp_pthreshold_sampling


def undirected_distance(structure, data):
    matrix = shortest_paths_from_struct(structure)
    data += matrix


def structure_to_network(structure):
    graph = nx.Graph()
    graph.add_nodes_from([x for x, _ in enumerate(structure)])
    pairtable = RNA.ptable(structure)[1:]
    for x, element in enumerate(pairtable):
        if x < len(pairtable) - 1:
            graph.add_edge(x, x+1)
            graph.add_edge(x + 1, x)
        if element > 0:
            graph.add_edge(x, element - 1)  # -1 because zero based
    return graph


def shortest_paths_from_struct(structure):
    graph = structure_to_network(structure)
    matrix = np.zeros((len(structure), len(structure)))
    for source in graph.nodes:
        d = nx.single_source_shortest_path_length(graph, source, cutoff=None)
        for key, value in d.items():
            matrix[source][key] = value
    return matrix


def sample_distance(sequence, nr_samples, md=None):
    sequence = str(sequence)
    if md is None:
        md = RNA.md()
    data = np.zeros((len(sequence), len(sequence)))
    # activate unique multibranch loop decomposition
    md.uniq_ML = 1
    # create fold compound object
    fc = RNA.fold_compound(sequence, md)

    # compute MFE
    (ss, mfe) = fc.mfe()

    # rescale Boltzmann factors according to MFE
    fc.exp_params_rescale(mfe)

    # compute partition function to fill DP matrices
    fc.pf()
    bppm = np.asarray(fc.bpp())[1:, 1:]
    i = fc.pbacktrack(nr_samples, rna_shortest_paths, data)
    data = data / i
    return data, bppm


def sample(sequence, nr_samples: int, md=None):
    """Samples structures for a sequence redundantly

    Draws :code:`nr_samples` structures and calculates the expected distance for all
    nucleotide pairs based on these.

    Args:
        sequence (str): RNA sequence as a string
        nr_samples (int): How many samples should be drawn
        md (RNA.md): ViennaRNA model details (Will automatically set uniq_ML to 1)

    Returns:
         np.ndarray : :code:`N x N` matrix
            containing approximated expected distances from nucleotide :code:`i` to :code:`j` at
            :code:`matrix[i][j]`
    """
    if md is None:
        md = RNA.md()
    md.uniq_ML = 1
    fc = RNA.fold_compound(sequence, md)
    return sample_fc(fc, nr_samples)


def sample_pthreshold(sequence, cutoff: float = 0.99, md=None):
    """Samples structures for a sequence non-redundantly

    Draws structures (:code:`S`)  via non redundant sampling from the ensemble of Structures (:code:`ES`)
    until the probability mass of structures reaches :code:`cutoff`
    Further calculates the expected distance (:code:`ED`) for all nucleotide pairs based on the following formula,
    where :code:`D` specifies the distance on a given structure :code:`S` and :code:`P(s)` is the corresponding
    probability.

    .. math::

        ED_{i, j} = \sum_{S \in SE} P(S) × D_{i, j}

    Args:
        sequence (str): RNA sequence as a string
        cutoff (float): Probability cutoff. If sum of probability of samples structures reaches this sampling stops
        md (RNA.md): ViennaRNA model details (Will automatically set uniq_ML to 1 and pf_smooth to 0)

    Returns:
         np.ndarray : :code:`N x N` matrix
            containing approximated expected distances from nucleotide :code:`i` to :code:`j` at
            :code:`matrix[i][j]`
    """
    if md is None:
        md = RNA.md()
    md.uniq_ML = 1
    md.pf_smooth = 0
    fc = RNA.fold_compound(sequence, md)
    return pthreshold_sample_fc(fc, cutoff)


def sample_fc(fc: RNA.fold_compound, nr_samples: int, undirected: bool = True):
    """Samples structures for a sequence and returns pairwise distances of nucleotides

    .. warning::

        This function might produce nonsense output if the fc is not set up correctly.
        If you do not know how to do this consider using
        :func:`~RNAdist.sampling.ed_sampling.sample`

    Args:
        fc (RNA.fold_compound): ViennaRNA fold compound.
        nr_samples (int): How many samples should be drawn
        undirected (bool): Whether to use undirected or directed Graph to calculate distances


    Returns:
         np.ndarray : :code:`N x N` matrix
            containing approximated expected distances from nucleotide :code:`i` to :code:`j` at
            :code:`matrix[i][j]`

    It is possible to sample expected distances using the ViennaRNA fold compound as follows. Please make sure
    to enable unique multiloop decomposition via :code:`uniq_ML=1`.

    >>> import RNA
    >>> seq = "GGGCUAUUAGCUCAGUUGGUUAGAGCGCACCCCUGAUAAGGGUGAGGUCGCUGAUUCGAAUUCAGCAUAGCCCA"
    >>> fc = RNA.fold_compound(seq, RNA.md(uniq_ML=1))
    >>> x = sample_fc(fc)
    >>> x[0, -1]
    2.004000000000001
    """
    fc.params.model_details.uniq_ML = 1
    return cpp_sampling(fc.this, nr_samples, undirected)


def pthreshold_sample_fc(fc: RNA.fold_compound, threshold: float = 0.99, undirected: bool = True):
    """Samples structures for a sequence non-redundantly and returns weighted pairwise nt distances.

    .. warning::

        This function might produce nonsense output if the fc is not set up correctly.
        If you do not know how to do this consider using
        :func:`~RNAdist.sampling.ed_sampling.sample_nr_cpp`

    Args:
        fc (RNA.fold_compound): ViennaRNA fold compound.
        threshold (float): Probability cutoff. If sum of probability of samples structures reaches this sampling stops
        undirected (bool): Whether to use undirected or directed Graph to calculate distances

    Returns:
         np.ndarray : :code:`N x N` matrix
            containing approximated expected distances from nucleotide :code:`i` to :code:`j` at
            :code:`matrix[i][j]`
    """
    assert 0 < threshold < 1, "Cutoff needs to be in range (0, 1)"
    assert fc.params.model_details.pf_smooth == 0, "PF smooth needs to be set to 0 for this mode please do so before" \
                                                   "filling the partition function"
    fc.params.model_details.uniq_ML = 1
    return cpp_pthreshold_sampling(fc.this, threshold, undirected)


def non_redundant_sample_fc(fc: RNA.fold_compound, nr_samples: int = 1, undirected: bool = True):
    """Samples structures for a sequence non-redundantly and returns pairwise distances of nucleotides

    .. warning::

        This function might produce nonsense output if the fc is not set up correctly.
        If you do not know how to do this consider using
        :func:`~RNAdist.sampling.ed_sampling.sample_nr_cpp`

    Args:
        fc (RNA.fold_compound): ViennaRNA fold compound.
        nr_samples (float): Number of non-redundant structures to draw
        undirected (bool): Whether to use undirected or directed Graph to calculate distances

    Returns:
         np.ndarray : :code:`N x N` matrix
            containing approximated expected distances from nucleotide :code:`i` to :code:`j` at
            :code:`matrix[i][j]`
    """
    assert fc.params.model_details.pf_smooth == 0, "PF smooth needs to be set to 0 for this mode please do so before" \
                                                   "filling the partition function"
    fc.params.model_details.uniq_ML = 1
    return cpp_nr_sampling(fc.this, nr_samples, undirected)


def rna_shortest_paths(s, data: List[np.ndarray]):
    """
    A simple callback function that adds shortest paths on structures
    to a N x N data matrix
    """
    if s:
        undirected_distance(s, data)
