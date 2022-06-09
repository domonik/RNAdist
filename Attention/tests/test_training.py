import Attention.training
from tempfile import TemporaryDirectory
import os
import torch
import subprocess
from Attention.tests.data_fixtures import (
    random_fasta,
    expected_labels,
    train_config,
    PREFIX
)


def test_main(random_fasta, train_config, expected_labels):
    with TemporaryDirectory(prefix=PREFIX) as tmpdir:
        Attention.training.main(
            fasta=random_fasta,
            label_dir=expected_labels,
            data_path=tmpdir,
            config=train_config,
            num_threads=os.cpu_count(),
            epochs=1,
            max_length=20,
            train_val_ratio=0.2
        )
        assert os.path.exists(train_config["model_checkpoint"])
        assert isinstance(torch.load(train_config["model_checkpoint"]), tuple)


def test_cmd_training(random_fasta, expected_labels, tmp_path):
    model_file = os.path.join(tmp_path, "cmd_model.pt")
    training_file = os.path.abspath(Attention.training.__file__)
    with TemporaryDirectory(prefix=PREFIX) as tmpdir:

        process = ["python", training_file,
                   "--input", random_fasta,
                   "--label_dir", expected_labels,
                   "--output", model_file,
                   "--data_path", tmpdir,
                   "--max_length", "20",
                   "--max_epochs", "1"
                   ]
        data = subprocess.run(process, stderr=subprocess.PIPE)
        assert data.stderr.decode() == ""
        assert os.path.exists(model_file)

