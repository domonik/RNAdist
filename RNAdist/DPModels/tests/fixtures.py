import pytest
import RNA


@pytest.fixture(scope="session")
def rna_dp_access():
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


@pytest.fixture(scope="session")
def seq4test():
    seq = "UUUCUCGCAAUGAUCAACGGGCAA"
    return seq