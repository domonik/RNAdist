import RNA
import numpy as np
from RNAdist.dp.cpp.RNAsProbs import cpp_struct_probs


def fold_bppm(sequence, md=None):
    if md is None:
        md = RNA.md()
    md.uniq_ML = 1
    # create fold compound object
    fc = RNA.fold_compound(sequence, md)

    # compute MFE
    ss, mfe = fc.mfe()

    # rescale Boltzmann factors according to MFE
    fc.exp_params_rescale(mfe)

    # compute partition function to fill DP matrices
    fc.pf()
    bppm = fc.bpp()
    bppm = np.asarray(bppm)[1:, 1:]
    bppm = bppm + bppm.T
    return bppm


def plfold_bppm(sequence, window, span, md=None):
    if md is None:
        md = RNA.md()
    md.max_bp_span = span
    md.window_size = window
    md.uniq_ML = 1
    bppm = np.zeros((len(sequence) + 1, len(sequence) + 1))
    fc = RNA.fold_compound(sequence, md, RNA.OPTION_WINDOW)
    fc.probs_window(1, RNA.PROBS_WINDOW_BPP, bpp_callback, bppm)
    bppm = np.asarray(bppm)
    bppm = bppm[1:, 1:]
    bppm[np.isnan(bppm)] = 0
    bppm = bppm + bppm.T
    return bppm


def bpp_callback(v, v_size, i, maxsize, what, data):
    if what:
        for j, p in enumerate(v):
            data[i, j] = p if p is not None else np.nan


def set_md_from_config(md, config):
    for key, value in config.items():
        setattr(md, key, value)


def _structural_probabilities(fc: RNA.fold_compound):
    # Different loop context probabilities.
    data_split = {'ext': [], 'hp': [], 'int': [], 'mb': []}

    # Get different loop context probabilities.
    fc.probs_window(1, RNA.PROBS_WINDOW_UP | RNA.PROBS_WINDOW_UP_SPLIT, up_split_callback, data_split)

    # Store individual probs for sequence in lists.
    ups = []
    ups_e = []
    ups_h = []
    ups_i = []
    ups_m = []
    ups_s = []
    for i in range(fc.length):
        p_e = 0
        p_h = 0
        p_i = 0
        p_m = 0
        if data_split['ext'][i]['up'][1]:
            p_e = data_split['ext'][i]['up'][1]
        if data_split['hp'][i]['up'][1]:
            p_h = data_split['hp'][i]['up'][1]
        if data_split['int'][i]['up'][1]:
            p_i = data_split['int'][i]['up'][1]
        if data_split['mb'][i]['up'][1]:
            p_m = data_split['mb'][i]['up'][1]
        # Total unpaired prob = sum of different loop context probs.
        p_u = p_e + p_h + p_i + p_m
        if p_u > 1:
            p_u = 1
        # Paired prob (stacked prob).
        p_s = 1 - p_u
        ups.append(p_u)
        ups_e.append(p_e)
        ups_h.append(p_h)
        ups_i.append(p_i)
        ups_m.append(p_m)
        ups_s.append(p_s)
    data = {
        "exterior": ups_e,
        "hairpin": ups_h,
        "interior": ups_i,
        "multiloop": ups_m,
        "idk": ups_s,
        "unpaired": ups
    }
    return data


def up_split_callback(v, v_size, i, maxsize, what, data):
    if what & RNA.PROBS_WINDOW_UP:
        what = what & ~RNA.PROBS_WINDOW_UP
        dat = []
        # Non-split case:
        if what == RNA.ANY_LOOP:
            dat = data
        # all the cases where probability is split into different loop contexts
        elif what == RNA.EXT_LOOP:
            dat = data['ext']
        elif what == RNA.HP_LOOP:
            dat = data['hp']
        elif what == RNA.INT_LOOP:
            dat = data['int']
        elif what == RNA.MB_LOOP:
            dat = data['mb']
        dat.append({'i': i, 'up': v})


def calc_struct_probs(fc: RNA.fold_compound):
    """ Calculates different structural probabilities

    Args:
        fc (RNA.fold_compound): ViennaRNA fold compound that is used for structural probability calculation


    Returns:
        :class:`RNAdist.dp.viennarna_helpers.RNAStructProbs`

    """
    array = cpp_struct_probs(fc.this)
    return RNAStructProbs(fc, array)


class RNAStructProbs:
    """Class that acts like a dictionary to store structural probabilities as a np.array

    An instance of this class is returned if you use the
    :func:`RNAdist.dp.viennarna_helpers.RNAStructProbs.calc_struct_probs` function.


    Args:
        fc (str): The viennaRNA fold compound the structural probabilities belong to
        array (np.ndarray): numpy array of shape (4, fc.length) storing structural probabilities

    It is possbible to access the different structural probabilities the following way:

    >>> sprobs: RNAStructProbs = calc_struct_probs(fc)
    >>> sprobs["exterior"]
    >>> sprobs["hairpin"]
    >>> sprobs["interior"]
    >>> sprobs["multiloop"]

    """

    _indices = {
        "exterior": 0,
        "hairpin": 1,
        "interior": 2,
        "multiloop": 3,
    }

    def __init__(self, fc: RNA.fold_compound, array: np.ndarray):
        self.fc = fc
        self.array = array

    def __getitem__(self, item):
        return self.array[self._indices[item]]
