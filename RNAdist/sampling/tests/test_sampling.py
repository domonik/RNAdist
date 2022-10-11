import pytest
from RNAdist.sampling.ed_sampling import sample_cpp, sample_nr_cpp
import RNA
import numpy as np


@pytest.mark.parametrize(
    "seq,temp",
    [
        ("AGCGCGCCUAAGACGCGCGAC", 37),
        ("AGCGCGCCUAAGACGCGCGAC", 20),
    ]
)
def test_redundant_cpp_samnpling(seq, temp):
    md = RNA.md(temperature=temp)
    result = sample_cpp(sequence=seq, nr_samples=10, md=md)
    assert np.isclose(result[0, 1], 1)


@pytest.mark.parametrize(
    "seq,temp,cutoff",
    [
        ("AGCGCGCCUAAGACGCGCGAC", 37, 0.99),
        ("AGCGCGCCUAAGACGCGCGAC", 20, 0.95),
    ]
)
def test_nr_cpp_samnpling(seq, temp, cutoff):
    md = RNA.md(temperature=temp)
    result = sample_nr_cpp(sequence=seq, cutoff=cutoff, md=md)
    assert np.greater_equal(result[0, 1], cutoff)