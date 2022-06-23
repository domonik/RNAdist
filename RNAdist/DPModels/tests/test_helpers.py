from RNAdist.DPModels.viennarna_helpers import (
    structural_probabilities,
    plfold_bppm
)
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
def test_plfold_bppm(seq4test, test_md):
    bppm = plfold_bppm(seq4test, len(seq4test), len(seq4test), test_md)
    assert not np.all(bppm == 0)
