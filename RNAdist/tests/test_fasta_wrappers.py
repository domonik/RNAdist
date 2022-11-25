import pandas as pd
import pytest
from RNAdist.fasta_wrappers import clote_ponty_from_fasta, pmcomp_from_fasta, sampled_distance_from_fasta, bed_distance_wrapper
from Bio import SeqIO


pytest_plugins = [
    "RNAdist.dp.tests.fixtures",
    "RNAdist.nn.tests.data_fixtures",
    "RNAdist.tests.fasta_fixtures"
]


@pytest.mark.parametrize(
    "function",
    [
        clote_ponty_from_fasta,
        pmcomp_from_fasta,
        sampled_distance_from_fasta
    ]
)
@pytest.mark.parametrize(
    "md_config,threads",
    [
        ({"temperature": 35}, 1),
        ({"temperature": 37}, 2),
    ]
)
def test_fasta_wrappers(random_fasta, md_config, threads, function):
    data = function(random_fasta, md_config, threads)
    for sr in SeqIO.parse(random_fasta, "fasta"):
        assert sr.description in data


@pytest.mark.parametrize(
    "beds,names",
    [
        (["bed_test_bed"], ["test_bed"])
    ]
)
@pytest.mark.parametrize(
    "md_config,threads",
    [
        ({"temperature": 35}, 1),
        ({"temperature": 37}, 2),
    ]
)
def test_binding_site_wrapper(bed_test_fasta, md_config, threads, beds, request, names):
    beds = [request.getfixturevalue(bed) for bed in beds]
    df = bed_distance_wrapper(bed_test_fasta, beds, md_config, names=names, num_threads=threads)
    assert df.shape[0] >= 1
    assert df.shape[1] == 8



