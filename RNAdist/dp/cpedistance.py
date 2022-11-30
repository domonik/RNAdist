import os
from contextlib import redirect_stdout
import numpy as np
import RNA
from CPExpectedDistance.p_expected_distance import expected_distance
from typing import List, Tuple
from tempfile import NamedTemporaryFile
from io import TextIOBase


def dp_matrix_available():
    """Checks whether you use a ViennaRNA version that supports DP matrix access

    Returns:
        bool: True if access to DP matrix is possible false else
    """
    seq = "AATATAT"
    md = RNA.md()
    md.uniq_ML = 1
    fc = RNA.fold_compound(seq, md)
    fc.pf()
    if hasattr(fc, 'exp_matrices') and hasattr(fc, 'exp_E_ext_stem'):
        dp_mx = fc.exp_matrices
        op = getattr(dp_mx, "get_Z", None)
        if callable(op):
            return True
    return False


def cp_expected_distance(sequence, md=None):
    """Calculate the expected Distance Matrix using the clote-ponty algorithm.

    Clote, P., Ponty, Y., & Steyaert, J. M. (2012).
    Expected distance between terminal nucleotides of RNA secondary structures.
    Journal of mathematical biology, 65(3), 581-599.

    https://doi.org/10.1007/s00285-011-0467-8

    Args:
        sequence (str): RNA sequence of size :code:`N`
        md (RNA.md): ViennaRNA model details object

    Returns:
        np.ndarray : :code:`N x N` matrix
            containing expected distance from nucleotide :code:`0` to :code:`n` at
            :code:`matrix[0][-1]`


    .. warning::

        Experimental function. might not work if your numpy version doesnt match the numpy version
        the C extension was compiled with

    You can calculate this using the default model details from ViennaRNA like this

    >>> seq = "GGGCUAUUAGCUCAGUUGGUUAGAGCGCACCCCUGAUAAGGGUGAGGUCGCUGAUUCGAAUUCAGCAUAGCCCA"
    >>> x = cp_expected_distance(seq)
    >>> x[0, -1]
    2.0040000903244186

    By including a model details object in the function call you can change settings for e.g.
    the temperature

    >>> seq = "GGGCUAUUAGCUCAGUUGGUUAGAGCGCACCCCUGAUAAGGGUGAGGUCGCUGAUUCGAAUUCAGCAUAGCCCA"
    >>> md = RNA.md(temperature=35.4)
    >>> x = cp_expected_distance(seq, md=md)
    >>> x[0, -1]
    2.0025985167453437

    """
    if md is None:
        md = RNA.md()
    md.uniq_ML = 1
    fc = RNA.fold_compound(sequence, md)
    return compute_clote(fc)


def compute_clote(fc):
    """Uses a ViennaRNA fold_compound to calculate the clote-ponty matrix.

    Clote, P., Ponty, Y., & Steyaert, J. M. (2012).
    Expected distance between terminal nucleotides of RNA secondary structures.
    Journal of mathematical biology, 65(3), 581-599.

    https://doi.org/10.1007/s00285-011-0467-8

    Args:
        fc (RNA.fold_compund): Fold compound object of ViennaRNA

    Returns:
        np.ndarray : :code:`N x N` matrix
            containing expected distance from nucleotide :code:`0` to :code:`n` at
            :code:`matrix[0][-1]`

    """
    return expected_distance(fc)


def binding_site_distance(fc: RNA.fold_compound, binding_sites: List[Tuple[int, int]]) -> float:
    """Calculates the expected distance between binding sites.

    Will force the binding sites to be in unstructured regions of the RNA (exterior or multi loop).
    Thus the expected distance using clote-ponty approach will be correct.

    See Also:
        :func:`~cp_expected_distance`, :func:`~compute_clote`

    Args:
        fc (RNA.fold_compund): Fold compound object of ViennaRNA
        binding_sites List(Tuple(int, int)): Two intervals of binding sites. (Zero based end non inclusive (BED style)

    Returns:
        float:
            expected distance between those binding sites (end of first, start of second)

    """
    binding_sites = sorted(binding_sites)
    _add_interval_constraints(fc, binding_sites)
    ed = expected_distance(fc)
    return float(ed[binding_sites[0][1]-1, binding_sites[1][0]])


def _add_interval_constraints(fc: RNA.fold_compound, binding_sites):
    with NamedTemporaryFile(mode="w+", suffix="RNAdist_commands") as tmpfile:
        _write_commands_to_file(binding_sites, fc.length, tmpfile)
        fc.file_commands_apply(tmpfile.name)
    return fc


def _write_commands_to_file(binding_sites: List[Tuple[int, int]], seq_len: int, file: TextIOBase):
    sites1 = _bed_to_vrna_interval(binding_sites[0])
    sites2 = _bed_to_vrna_interval(binding_sites[1])
    assert sites1[1] < sites2[0], "This method does not support overlapping sites"
    if sites1[0] > 1:
        # permits spanning from before to back
        file.write(f"P 1-{sites1[0]-1} {sites1[1]+1}-{seq_len}\n")
    if sites2[1] < seq_len:
        # permits span f behind to front
        file.write(f"P 1-{sites2[0]-1} {sites2[1]+1}-{seq_len}\n")
    file.write(f"P {sites1[0]} 0 {sites1[1]- sites1[0] + 1}\n")
    file.write(f"P {sites2[0]} 0 {sites2[1] - sites2[0] + 1}\n")
    file.seek(0)


def _bed_to_vrna_interval(bed_interval: Tuple[int, int]):
    interval = (bed_interval[0]+1, bed_interval[1])
    return interval
