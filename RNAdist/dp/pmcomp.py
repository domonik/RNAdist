import numpy as np
import RNA
from RNAdist.dp._dp_calculations import _fast_pmcomp


def pmcomp_distance(sequence, md=None):
    """Approximates Expected Distances using basepair probabilities

    Calculates Approximated Expected Distances using the following formulae:

    .. math::

        jupin(i, j) = 1 - \sum_{i < k < j} p_{k,j}

        E_{i,j} = jupin(i, j) * (E_{i,j-1}+1) + p_{i,j} + \sum_{i < k < j} (E_{i,k-1}+2) * p_{k,j}

    Args:
        sequence (str): RNA sequence of size :code:`N`
        md (RNA.md): ViennaRNA model details object

    Returns:
         np.ndarray : :code:`N x N` matrix
            containing approximated expected distances from nucleotide :code:`i` to :code:`j` at
            :code:`matrix[i][j]`

    You can calculate this using the default model details from ViennaRNA like this

    >>> seq = "GGGCUAUUAGCUCAGUUGGUUAGAGCGCACCCCUGAUAAGGGUGAGGUCGCUGAUUCGAAUUCAGCAUAGCCCA"
    >>> x = pmcomp_distance(seq)
    >>> x.shape
    (74, 74)

    Via including a model details object in the function call you can change settings for e.g.
    the temperature

    >>> seq = "GGGCUAUUAGCUCAGUUGGUUAGAGCGCACCCCUGAUAAGGGUGAGGUCGCUGAUUCGAAUUCAGCAUAGCCCA"
    >>> md = RNA.md(temperature=35.4)
    >>> x = pmcomp_distance(seq, md=md)
    >>> x[:5, -5:]
    array([[4.00170356, 3.00188247, 2.00211336, 1.00316215, 2.0033279 ],
       [3.00170363, 2.00188257, 1.00212447, 2.00234503, 3.00249076],
       [2.00170369, 1.00188828, 2.00369139, 3.00389263, 4.00401832],
       [1.00170376, 2.00329739, 3.00485005, 4.00502225, 5.0051279 ],
       [2.00676375, 3.00810301, 4.00939988, 5.00954299, 6.00962855]])

    """
    if md is None:
        md = RNA.md()
    md.uniq_ML = 1
    fc = RNA.fold_compound(sequence, md)
    (ss, mfe) = fc.mfe()
    fc.exp_params_rescale(mfe)
    fc.pf()
    expected_d = pmcomp_dist_from_fc(fc)
    return expected_d


def pmcomp_dist_from_fc(fc):
    """Approximates Expected Distances using basepair probabilities from the ViennaRNA fold_compound

    .. warning::

        This function might produce nonsense output if the fc is not set up correctly.
        If you do not know how to do this consider using
        :func:`~RNAdist.dp.pmcomp.pmcomp_distance`

    Args:
        fc (RNA.fold_compund): Fold compound object of ViennaRNA

    Returns:
        np.ndarray : :code:`N x N` matrix
            containing expected distance from nucleotide :code:`0` to :code:`n` at
            :code:`matrix[0][-1]`
    Raises:
        RuntimeError: If the DP matrices are not filled yet due to a missing fc.pf() call

    """
    return _fast_pmcomp(fc)


