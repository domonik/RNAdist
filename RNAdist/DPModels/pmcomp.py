import numpy as np
import RNA


# def upprob_and_bppm(seq):
#     bppm = fold_bppm(seq)
#     upprob = 1 - bppm.sum(axis=1)
#     return upprob.tolist(), bppm.numpy()
#
#
# def up_from_bppm(bppm):
#     return 1 - bppm.sum(axis=1)

def _up_in_ij(bppm, min_len: int = 3):
    size = bppm.shape[0]
    up_in = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            for k in range(i, j - min_len + 1):
                up_in[i, j] += bppm[k, j]
    return 1 - up_in


def _init(bppm, min_len: int = 3):
    m = np.zeros(bppm.shape)
    for x in range(min_len):
        dia = np.ones(bppm.shape[0] - (x + 1)) * (x+1)
        new_m = np.diag(dia, x + 1)
        m += new_m
    return m


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
        :func:`~RNAdist.DPModels.pmcomp.pmcomp_distance`

    Args:
        fc (RNA.fold_compund): Fold compound object of ViennaRNA

    Returns:
        np.ndarray : :code:`N x N` matrix
            containing expected distance from nucleotide :code:`0` to :code:`n` at
            :code:`matrix[0][-1]`
    Raises:
        RuntimeError: If the DP matrices are not filled yet due to a missing fc.pf() call

    """
    min_len = fc.params.model_details.min_loop_size
    try:
        bppm = np.asarray(fc.bpp())[1:, 1:]
    except IndexError:
        raise RuntimeError("DP matrices have no been filled yet. "
                           "Please call the pf() function of the fold compound first")
    up_in = _up_in_ij(bppm)
    expected_d = _init(bppm, min_len)
    n = bppm.shape[0]
    for i in range(n):
        for j in range(i + min_len + 1, n):
            subterm = 0
            debug = {}
            de2 = 0
            for k in range(i + 1, j - min_len + 1):
                subterm += (expected_d[i, k - 1] + 2) * bppm[k, j]
                if (expected_d[i, k - 1] + 1) * bppm[k, j] != 0:
                    debug[k] = (expected_d[i, k - 1] + 2) * bppm[k, j], expected_d[i, k - 1], bppm[k, j]
                    de2 += bppm[k, j]
            expected_d[i, j] = (up_in[i, j] * (expected_d[i, j - 1] + 1)) + subterm + bppm[i, j]
            de2 += up_in[i, j]
    return expected_d


if __name__ == '__main__':
    sequence = "UCCCAACAGGGCUGUUCCCGGUUGGCCGGCGACCUUAGGGUUAAGCCUA"
    sequence = sequence.replace("T", "U")
    d = pmcomp_distance(sequence)
    p = 0


