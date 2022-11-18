import pytest
from RNAdist.sampling.ed_sampling import sample, sample_pthreshold, non_redundant_sample_fc
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
