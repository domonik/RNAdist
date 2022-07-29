import pytest
from RNAdist.fasta_wrappers import clote_ponty_from_fasta, pmcomp_from_fasta, sampled_distance_from_fasta
from Bio import SeqIO


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
