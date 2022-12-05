from RNAdist.dp.viennarna_helpers import structural_probabilities, fold_bppm, plfold_bppm
import pytest
import RNA
import numpy as np

pytest_plugins = ["RNAdist.dp.tests.fixtures",
                  "RNAdist.nn.tests.data_fixtures"]

@pytest.mark.parametrize(
    "test_md",
    (
        [None, RNA.md()]
    )
)
def test_structural_probabilities(seq4test, test_md):
    fc = RNA.fold_compound(seq4test, test_md)
    probabilities = structural_probabilities(fc)
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
    assert np.all(np.triu(bppm) == np.tril(bppm).T)


@pytest.mark.parametrize(
    "test_md",
    (
        [None, RNA.md()]
    )
)
def test_fold_bppm(seq4test, test_md):
    bppm = fold_bppm(seq4test, test_md)
    assert np.all(np.triu(bppm) == np.tril(bppm).T)
