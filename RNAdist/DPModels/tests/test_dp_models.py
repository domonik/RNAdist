import pytest
import RNA
from RNAdist.DPModels.clote import cp_expected_distance


@pytest.mark.parametrize(
    "test_md",
    (
        [None, RNA.md()]
    )
)
def test_clote_ponty(seq4test, test_md, rna_dp_access):
    print(RNA.__version__)
    if not rna_dp_access:
        pytest.skip("Your ViennaRNA Python API does not allow DP access")
    cp_expected_distance(sequence=seq4test, md=test_md)

