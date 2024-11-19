import pytest
from RNAdist.sampling.ed_sampling import sample, sample_pthreshold, non_redundant_sample_fc, sample_distance_ij
import RNA
import numpy as np


@pytest.mark.parametrize(
    "seq,temp",
    [
        ("AGCGCGCCUAAGACGCGCGAC", 37),
        ("AGCGCGCCUAAGACGCGCGAC", 20),
    ]
)
def test_redundant_cpp_sampling(seq, temp):
    md = RNA.md(temperature=temp)
    result = sample(sequence=seq, nr_samples=10, md=md)
    assert np.isclose(result[0, 1], 1)


@pytest.mark.parametrize(
    "seq,temp",
    [
        ("AGCGCGCCUAAGACGCGCGAC", 37),
        ("AGCGCGCCUAAGACGCGCGAC", 20),
    ]
)
def test_non_redundant_cpp_sampling(seq, temp):
    md = RNA.md(temperature=temp, pf_smooth=0)
    fc = RNA.fold_compound(seq, md)
    result = non_redundant_sample_fc(fc, nr_samples=10)
    assert result[0][1] != 0


@pytest.mark.parametrize(
    "seq,temp,cutoff",
    [
        ("AGCGCGCCUAAGACGCGCGAC", 37, 0.99),
        ("AGCGCGCCUAAGACGCGCGAC", 20, 0.95),
    ]
)
def test_threshold_cpp_sampling(seq, temp, cutoff):
    md = RNA.md(temperature=temp)
    result = sample_pthreshold(sequence=seq, cutoff=cutoff, md=md)
    assert np.greater_equal(result[0, 1], cutoff)


@pytest.mark.parametrize(
    "seq,temp",
    [
        ("AGCGCGCCUAAGACGCGCGAC", 37),
        ("AGCGCGCCUAAGACGCGCGAC", 20),
    ]
)
def test_ij_sampling(seq, temp ):
    md = RNA.md(temperature=temp)
    fc = RNA.fold_compound(seq, md)
    ed = np.zeros((len(seq), len(seq)))
    for i in range(len(seq)):
        for j in range(i, len(seq)):
            ed[i, j] = sample_distance_ij(fc, i, j, nr_samples=1000)
    ed = ed + ed.T
    result = sample(sequence=seq, nr_samples=1000, md=md)
    assert np.allclose(result, ed, atol=0.2)



