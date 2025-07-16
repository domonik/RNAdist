import pytest
import RNA
from RNAdist.dp.cpedistance import dp_matrix_available


@pytest.fixture(scope="session")
def rna_dp_access():
    return dp_matrix_available()


@pytest.fixture(scope="session")
def seq4test():
    seq = "UUUCUCGCAAUGAUCAACGGGCAA"
    return seq