from RNAdist.dp.viennarna_helpers import _structural_probabilities, fold_bppm, plfold_bppm
from RNAdist.dp.viennarna_helpers import calc_struct_probs
import pytest
import RNA
import numpy as np

pytest_plugins = ["RNAdist.dp.tests.fixtures"]


@pytest.mark.parametrize(
    "test_md",
    (
        [None, RNA.md()]
    )
)
def test_structural_probabilities(seq4test, test_md):
    fc = RNA.fold_compound(seq4test, test_md)
    probabilities = _structural_probabilities(fc)
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


def test_cpp_struct_probs(seq4test):
    md = RNA.md()
    plfold_l = plfold_w = len(seq4test)
    md.max_bp_span = plfold_l
    md.window_size = plfold_w
    fc = RNA.fold_compound(seq4test, md)
    sprobscpp = calc_struct_probs(fc)
    contexts = sprobscpp._indices.keys()

    sprobs = _structural_probabilities(fc)
    for context in contexts:
        np.equal(sprobscpp[context], np.asarray(sprobs[context]))



