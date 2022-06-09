import os
from tempfile import TemporaryDirectory, NamedTemporaryFile
from Attention.tests.data_fixtures import (
    random_fasta,
    generation_config,
    expected_labels,
    PREFIX
)
import pytest

from Attention.training_set_generation import create_random_fasta, \
    training_set_from_fasta, LabelDict


def test_random_fasta_generation(random_fasta):
    with open(random_fasta) as handle:
        expected = handle.read()
    with NamedTemporaryFile(prefix=PREFIX, mode="w+") as tmpfile:
        create_random_fasta(
            outpath=tmpfile.name,
            nr_sequences=10,
            seqrange=(10, 20),
            seed=0
        )
        tmpfile.seek(0)
        actual = tmpfile.read()
    assert actual == expected


def test_dataset_from_fasta(random_fasta, generation_config, expected_labels):
    expected = LabelDict(expected_labels)
    with TemporaryDirectory(prefix=PREFIX) as tmpdir:
        training_set_from_fasta(
            random_fasta,
            tmpdir,
            generation_config,
            num_threads=os.cpu_count(),
            bin_size=1,
            nr_samples=1
        )
        actual = LabelDict(tmpdir)
        for key, _ in expected.items():
            assert key in actual

