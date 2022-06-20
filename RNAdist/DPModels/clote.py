#!/usr/bin/env python3
#
import numpy as np
import RNA


# below is the Clote & Ponty DP algorithm

def cp_expected_distance(sequence, md=None):
    if md is None:
        md = RNA.md()
    md.uniq_ML = 1
    fc = RNA.fold_compound(sequence, md)
    (ss, mfe) = fc.mfe()

    fc.exp_params_rescale(mfe)

    fc.pf()
    return compute_clote(sequence, fc)


def compute_clote(sequence, fc):
    dp_mx = fc.exp_matrices
    output_matrix = np.zeros((len(sequence)+1, len(sequence)+1))

    # compute this weird D(i,j) of Clote et al. 2012
    for l in range(1, len(sequence) + 1):
        for i in range(1, len(sequence) + 1 - l):
            j = i + l
            #print("d({},{}) = {} vs. {}".format(i, j, output_matrix[i][j] * dp_mx.get_scale(1), j - i))
            dist = output_matrix[i][j - 1] * dp_mx.get_scale(1) + dp_mx.get_Z(i, j)
            for k in range(i + 1, j + 1):
                dist += (output_matrix[i][k - 1] + dp_mx.get_Z(i, k - 1)) * dp_mx.get_ZB(k, j) * fc.exp_E_ext_stem(k, j)
            output_matrix[i][j] = dist
    # now divide everything by Z(i,j) to obtain actual expected distances
    for i in range(1, len(sequence) + 1):
        for j in range(i + 1, len(sequence) + 1):
            output_matrix[i][j] /= dp_mx.get_Z(i, j)
            #print("d({},{}) = {} vs. {}".format(i, j, output_matrix[i][j], j - i))
    output_matrix = output_matrix[1:, 1:]
    return output_matrix



if __name__ == '__main__':
    seq = "AAAGGGAAACCCAAA"
    cp_expected_distance(seq)
