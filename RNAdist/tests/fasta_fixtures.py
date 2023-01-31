import os
import pickle

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


@pytest.fixture
def example_output_path():
    eop = os.path.join(TESTDATADIR, "example_output.pckl")
    assert os.path.exists(eop)
    return eop

@pytest.fixture
def example_output(example_output_path):
    with open(example_output_path, "rb") as handle:
        eop = pickle.load(handle)
    return eop