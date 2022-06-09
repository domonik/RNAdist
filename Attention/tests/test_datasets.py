from Attention.Datasets import RNAPairDataset
from tempfile import TemporaryDirectory
from Attention.tests.data_fixtures import (
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
