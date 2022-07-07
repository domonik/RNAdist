import RNA
import numpy as np


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


def set_md_from_config(md, config):
    for key, value in config.items():
        setattr(md, key, value)


def structural_probabilities(sequence, md=None):
    """
    Calculate structural elements probabilities (different loop contexts),
    using ViennaRNA's RNAplfold.
    This uses the Python3 API (RNA.py) of ViennaRNA (tested with v 2.4.17).
    So RNA.py needs to be in PYTHONPATH, which it is,
    if e.g. installed via:
    conda install -c bioconda viennarna=2.4.17
    sequence:
        Input sequence as string
    returns:
        data: Dict[str, List[float]]
            Dictionary with structural context probabilities
    """


    # Check input.

    # Output files.
    # Calculate base pair and structural elements probabilities.
    plfold_l = plfold_w = len(sequence)
    if md is None:
        md = RNA.md()
    md.max_bp_span = plfold_l
    md.window_size = plfold_w

    # Different loop context probabilities.
    data_split = {'ext': [], 'hp': [], 'int': [], 'mb': [] }

    fc = RNA.fold_compound(sequence, md, RNA.OPTION_WINDOW)
    # Get different loop context probabilities.
    fc.probs_window(1, RNA.PROBS_WINDOW_UP | RNA.PROBS_WINDOW_UP_SPLIT, up_split_callback, data_split)

    # Store individual probs for sequence in lists.
    ups = []
    ups_e = []
    ups_h = []
    ups_i = []
    ups_m = []
    ups_s = []

    for i,e in enumerate(sequence):
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
    """
    This uses the Python3 API (RNA.py) of ViennaRNA (tested with v 2.4.13).
    So RNA.py needs to be in PYTHONPATH (it is if installed via conda).
    """
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