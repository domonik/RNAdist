from Attention.training import main
from tempfile import TemporaryDirectory
import os
import torch
from Attention.tests.data_fixtures import (
    random_fasta,
    expected_labels,
    train_config,
    PREFIX
)


def test_main(random_fasta, train_config, expected_labels):
    with TemporaryDirectory(prefix=PREFIX) as tmpdir:
        main(
            fasta=random_fasta,
            label_dir=expected_labels,
            data_path=tmpdir,
            config=train_config,
            num_threads=1,
            epochs=1,
            max_length=20,
            train_val_ratio=0.2
        )
        assert os.path.exists(train_config["model_checkpoint"])
        assert isinstance(torch.load(train_config["model_checkpoint"]), tuple)
