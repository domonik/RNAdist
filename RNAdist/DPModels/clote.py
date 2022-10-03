import numpy as np
import RNA
from CPExpectedDistance.p_expected_distance import expected_distance


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

    .. warning::

        This function is deprecated and only here for backward compatibility.
        It uses the ViennaRNA C API to create a numpy array.
        However, it messes around with the representation of the fold compound. consider using:

        :func:`~RNAdist.DPModels.clote.cp_expected_distance`


    Args:
        fc (RNA.fold_compund): Fold compound object of ViennaRNA

    Returns:
        np.ndarray : :code:`N x N` matrix
            containing expected distance from nucleotide :code:`0` to :code:`n` at
            :code:`matrix[0][-1]`
    Raises:
        RuntimeError: If the DP matrices are not filled yet due to a missing fc.pf() call

    """
    return expected_distance(fc)
