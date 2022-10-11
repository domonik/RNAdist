from RNAdist.NNModels.Datasets import (
    RNAPairDataset,
    RNADATA,
    RNAWindowDataset
)
from tempfile import TemporaryDirectory
import torch
from RNAdist.NNModels.tests.data_fixtures import (
    random_fasta,
    expected_labels,
    PREFIX
)

pytest_plugins = ["RNAdist.DPModels.tests.fixtures",
                  "RNAdist.NNModels.tests.data_fixtures"]

def test_rna_pair_dataset(random_fasta, expected_labels):
    with TemporaryDirectory(prefix=PREFIX) as tmpdir:
        _ = RNAPairDataset(
            data=random_fasta,
            label_dir=expected_labels,
            dataset_path=tmpdir,
            num_threads=1,
            max_length=100
        )


def test_rna_window_dataset(random_fasta, expected_labels):
    ml = 9
    with TemporaryDirectory(prefix=PREFIX) as tmpdir:
        dataset = RNAWindowDataset(
            data=random_fasta,
            label_dir=None,
            dataset_path=tmpdir,
            num_threads=1,
            max_length=ml,
            step_size=1
        )
        for x in range(len(dataset)):
            pair_matrix, y, mask, indices = dataset[x]
            assert pair_matrix.shape[0] == ml
            assert pair_matrix.shape[1] == ml
            assert not torch.any(torch.isnan(pair_matrix))


def test_rna_data_to_tensor(seq4test, expected_rna_data):
    rna_data = RNADATA(seq4test, "foo")
    pair, x = rna_data.to_tensor()
    expected_pair, expected_x = expected_rna_data
    assert torch.equal(expected_pair, pair)
    assert torch.equal(expected_x, x)