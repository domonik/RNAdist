import os
from tempfile import TemporaryDirectory, NamedTemporaryFile
import pytest

from RNAdist.nn.training_set_generation import create_random_fasta, \
    training_set_from_fasta, LabelDict

pytest_plugins = ["RNAdist.dp.tests.fixtures",
                  "RNAdist.nn.tests.data_fixtures"]

def test_random_fasta_generation(random_fasta, prefix):
    with open(random_fasta) as handle:
        expected = handle.read()
    with NamedTemporaryFile(prefix=prefix, mode="w+") as tmpfile:
        create_random_fasta(
            outpath=tmpfile.name,
            nr_sequences=10,
            seqrange=(10, 20),
            seed=0
        )
        tmpfile.seek(0)
        actual = tmpfile.read()
    assert actual == expected


def test_window_set_from_fasta(random_fasta, window_config, expected_window_labels, prefix):
    expected = LabelDict(expected_window_labels)
    with TemporaryDirectory(prefix=prefix) as tmpdir:
        training_set_from_fasta(
            random_fasta,
            tmpdir,
            window_config,
            num_threads=1,
            bin_size=1,
            nr_samples=1
        )
        actual = LabelDict(tmpdir)
        for key, _ in expected.items():
            assert key in actual


def test_dataset_from_fasta(random_fasta, generation_config, expected_labels, prefix):
    expected = LabelDict(expected_labels)
    with TemporaryDirectory(prefix=prefix) as tmpdir:
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

