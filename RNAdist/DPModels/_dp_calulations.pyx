import numpy as np


cdef _up_in_ij(double[:, :] bppm, int min_len = 3):
    cdef Py_ssize_t size = bppm.shape[0]
    up_in = np.zeros((size, size), dtype=np.float64)
    cdef double[:, :] up_view = up_in
    cdef Py_ssize_t i, j, k
    for i in range(size):
        for j in range(size):
            for k in range(i, j - min_len + 1):
                up_view[i, j] += bppm[k, j]
    return 1 - up_in



def _init(int size, int min_len = 3):
    m = np.zeros((size, size), dtype=np.float64)
    cdef double[:, :] m_view = m
    cdef int x, y
    for x in range(0, size):
        for y in range(1, min_len+1):
            if y+x < size:
                m_view[x, y+x] = y
    return m


def _fast_pmcomp(fc):
    cdef int min_len = fc.params.model_details.min_loop_size
    try:
        bppm = np.asarray(fc.bpp())[1:, 1:]
    except IndexError:
        raise RuntimeError("DP matrices have no been filled yet. "
                           "Please call the pf() function of the fold compound first")
    up_in = _up_in_ij(bppm)
    expected_d = _init(bppm.shape[0], min_len)
    cdef double[:, :] expected_d_view = expected_d
    cdef double[:, :] up_view = up_in
    cdef double[:, :] bppm_view = bppm

    cdef Py_ssize_t n = bppm.shape[0]
    cdef Py_ssize_t i, j, k
    cdef double subterm
    for i in range(n):
        for j in range(i + min_len + 1, n):
            subterm = 0
            for k in range(i + 1, j - min_len + 1):
                subterm += (expected_d_view[i, k - 1] + 2) * bppm_view[k, j]
            expected_d_view[i, j] = (up_view[i, j] * (expected_d_view[i, j - 1] + 1)) + subterm + bppm_view[i, j]
    return expected_d


def _fast_clote_ponty(fc):
    dp_mx = fc.exp_matrices
    if dp_mx is None:
        raise RuntimeError("DP matrices have not been filled yet. "
                           "Please call the pf() function of the fold compound first")
    output_matrix = np.zeros((fc.length + 1, fc.length + 1), dtype=np.float64)
    cdef double[:, :] output_view = output_matrix

    # compute D(i,j) of Clote et al. 2012
    cdef Py_ssize_t i, j, k
    cdef double dist
    for l in range(1, fc.length + 1):
        for i in range(1, fc.length + 1 - l):
            j = i + l
            dist = output_view[i][j - 1] * dp_mx.get_scale(1) + dp_mx.get_Z(i, j)
            for k in range(i + 1, j + 1):
                dist += (output_view[i][k - 1] + dp_mx.get_Z(i, k - 1)) * dp_mx.get_ZB(k, j) * fc.exp_E_ext_stem(k, j)
            output_view[i][j] = dist
    # now divide everything by Z(i,j) to obtain actual expected distances
    for i in range(1, fc.length + 1):
        for j in range(i + 1, fc.length + 1):
            output_view[i][j] /= dp_mx.get_Z(i, j)
    output_matrix = output_matrix[1:, 1:]
    return output_matrix