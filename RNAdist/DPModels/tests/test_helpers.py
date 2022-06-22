from RNAdist.DPModels.viennarna_helpers import structural_probabilities
import pytest
import RNA


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
