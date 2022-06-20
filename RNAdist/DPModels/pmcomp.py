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

def up_in_ij(bppm, min_len: int = 3):
    size = bppm.shape[0]
    up_in = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            for k in range(i, j - min_len + 1):
                up_in[i, j] += bppm[k, j]
    return 1 - up_in


def init(bppm, min_len: int = 3):
    m = np.zeros(bppm.shape)
    for x in range(min_len):
        dia = np.ones(bppm.shape[0] - (x + 1)) * (x+1)
        new_m = np.diag(dia, x + 1)
        m += new_m
    return m


def pmcomp_distance(sequence, md=None):
    if md is None:
        md = RNA.md()
    md.uniq_ML = 1
    fc = RNA.fold_compound(sequence, md)
    (ss, mfe) = fc.mfe()
    fc.exp_params_rescale(mfe)
    fc.pf()
    expected_d = pmcomp_dist_from_fc(fc, md.min_loop_size)
    return expected_d


def pmcomp_dist_from_fc(fc, min_len):
    bppm = np.asarray(fc.bpp())[1:, 1:]
    up_in = up_in_ij(bppm)
    expected_d = init(bppm, min_len)
    n = bppm.shape[0]
    for i in range(n):
        for j in range(i + min_len + 1, n):
            subterm = 0
            debug = {}
            de2 = 0
            if j == 16:
                p = 0
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


