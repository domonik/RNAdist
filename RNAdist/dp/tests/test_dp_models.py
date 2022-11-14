import pytest
import RNA
from RNAdist.dp.cpedistance import cp_expected_distance, _add_interval_constraints, binding_site_distance
from RNAdist.dp.pmcomp import pmcomp_distance
import numpy as np

pytest_plugins = ["RNAdist.dp.tests.fixtures",
                  "RNAdist.nn.tests.data_fixtures"]

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


@pytest.mark.parametrize(
    "intervals,seq",
    [
        ([(3, 8), (15, 17)], "AAUCAGUCUGAGUCAGUCAGUCUGUGCACUGA"),
        ([(17, 19), (30, 34)], "AGUGUGGGGCCGCGGCGCGUAUGCUAGCUGAUGAUGCUGCAUG"),
        ([(0, 5), (30, 34)], "AGUGUGGGGCCGCGGCGCGUAUGCUAGCUGAUGAUGCUGCAUG"),
        ([(5, 20), (43, 43)], "AGUGUGGGGCCGCGGCGCGUAUGCUAGCUGAUGAUGCUGCAUG"),
    ]
)
def test_interval_span_constraints(intervals, seq):
    md = RNA.md()
    md.uniq_ML = 1
    fc = RNA.fold_compound(seq, md)
    intervals = sorted(intervals)
    inner = intervals[0][0], intervals[1][1]
    _add_interval_constraints(fc, intervals)
    fc.pf()
    bpp = np.asarray(fc.bpp())[1:, 1:]
    front = bpp[0:intervals[0][0], inner[0]:inner[1]]
    back = bpp[inner[0]:inner[1], intervals[1][0]:]
    assert np.all(front == 0)
    assert np.all(back == 0)


@pytest.mark.parametrize(
    "intervals,seq",
    [
        ([(3, 8), (5, 17)], "AAUCAGUCUGAGUCAGUCAGUCUGUGCACUGA"),
    ]
)
def test_interval_span_fail(intervals, seq):
    md = RNA.md()
    fc = RNA.fold_compound(seq, md)
    with pytest.raises(AssertionError) as error:
        _add_interval_constraints(fc, intervals)
    assert str(error.value) == "This method does not support overlapping sites"


@pytest.mark.parametrize(
    "intervals,seq",
    [
        ([(3, 8), (15, 17)], "AAUCAGUCUGAGUCAGUCAGUCUGUGCACUGA"),
        ([(17, 19), (30, 17)], "AGUGUGGGGCCGCGGCGCGUAUGCUAGCUGAUGAUGCUGCAUG"),
    ]
)
def test_binding_site_distance(intervals, seq):
    """Tests whether its possible to use the cp distance approach for binding sites.
    WARNING: Does not test for correct results
    """
    fc = RNA.fold_compound(seq)
    ed = binding_site_distance(fc, intervals)
    assert ed != 0
