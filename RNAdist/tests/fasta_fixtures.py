import os
import pytest

TESTDATADIR = os.path.join(os.path.dirname(__file__), "test_data")


@pytest.fixture
def bed_test_bed():
    test_bed = os.path.join(TESTDATADIR, "bed_test.bed")
    assert os.path.exists(test_bed)
    return test_bed


@pytest.fixture
def bed_test_fasta():
    test_fa = os.path.join(TESTDATADIR, "bed_test.fa")
    assert os.path.exists(test_fa)
    return test_fa