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
    bppm = np.asarray(bppm)
    return bppm[1:, 1:]