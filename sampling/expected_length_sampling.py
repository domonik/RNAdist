import RNA
import numpy as np
from tempfile import NamedTemporaryFile
import subprocess
import os
import networkx as nx
from typing import List, Dict, Iterable


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
    #md.dangles = 0
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


def rna_shortest_paths(s, data: List[np.ndarray]):
    """
    A simple callback function that adds shortest paths on structures
    to a N x N data matrix
    """
    if s:
        undirected_distance(s, data)


if __name__ == '__main__':
    pass
