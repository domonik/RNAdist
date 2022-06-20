import pytest
import RNA
from RNAdist.DPModels.clote import cp_expected_distance
from RNAdist.DPModels.pmcomp import pmcomp_distance
import numpy as np

@pytest.mark.parametrize(
    "test_md",
    (
        [None, RNA.md()]
    )
)
def test_clote_ponty(seq4test, test_md, rna_dp_access):
    """ Tests whether its possible to use the cp approach.
    WARNING: Does not test for correct results
    """
    if not rna_dp_access:
        pytest.skip(
            "Your ViennaRNA Python API does not allow DP matrix access"
        )
    exp_d = cp_expected_distance(sequence=seq4test, md=test_md)
    assert isinstance(exp_d, np.ndarray)


@pytest.mark.parametrize(
    "test_md",
    (
        [None, RNA.md()]
    )
)
def test_pmcomp(seq4test, test_md):
    """ Tests whether its possible to use the pmcomp distance approach.
    WARNING: Does not test for correct results
    """
    distances = pmcomp_distance(seq4test, md=test_md)
    assert isinstance(distances, np.ndarray)
