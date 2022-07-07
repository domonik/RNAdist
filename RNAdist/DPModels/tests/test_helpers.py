from RNAdist.DPModels.viennarna_helpers import structural_probabilities, fold_bppm
import pytest
import RNA
import numpy as np


@pytest.mark.parametrize(
    "test_md",
    (
        [None, RNA.md()]
    )
)
def test_structural_probabilities(seq4test, test_md):
    probabilities = structural_probabilities(seq4test, test_md)
    assert isinstance(probabilities, dict)
    assert "exterior" in probabilities

@pytest.mark.parametrize(
    "test_md",
    (
        [None, RNA.md()]
    )
)
def test_fold_bppm(seq4test, test_md):
    bppm = fold_bppm(seq4test, test_md)
    assert np.all(np.triu(bppm) == np.tril(bppm).T)
