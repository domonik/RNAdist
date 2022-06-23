from RNAdist.Attention.Datasets import RNAPairDataset, RNADATA
from tempfile import TemporaryDirectory
import torch
from RNAdist.Attention.tests.data_fixtures import (
    random_fasta,
    expected_labels,
    PREFIX
)


def test_rna_pair_dataset(random_fasta, expected_labels):
    with TemporaryDirectory(prefix=PREFIX) as tmpdir:
        _ = RNAPairDataset(
            data=random_fasta,
            label_dir=expected_labels,
            dataset_path=tmpdir,
            num_threads=1,
            max_length=100
        )


def test_rna_data_to_tensor(seq4test, expected_rna_data):
    rna_data = RNADATA(seq4test, "foo")
    pair, x = rna_data.to_tensor(max_length=40, positional_encoding=True)
    expected_pair, expected_x = expected_rna_data
    assert torch.equal(expected_pair, pair)
    assert torch.equal(expected_x, x)