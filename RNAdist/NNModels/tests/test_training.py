from RNAdist.NNModels.training import train_network
from tempfile import TemporaryDirectory
import executables
import os
import torch
import subprocess
from RNAdist.NNModels.tests.data_fixtures import (
    random_fasta,
    expected_labels,
    train_config,
    PREFIX
)
import pytest


@pytest.mark.parametrize("model_type", ["normal", "small"])
@pytest.mark.parametrize("mode", ["normal", "window"])
def test_training(random_fasta, train_config, expected_labels, model_type, expected_window_labels, mode):
    train_config["model"] = model_type
    if mode == "normal":
        expected_labels = expected_labels
        ml = 20
    else:
        expected_labels = expected_window_labels
        ml = 10
    with TemporaryDirectory(prefix=PREFIX) as tmpdir:
        train_network(
            fasta=random_fasta,
            label_dir=expected_labels,
            dataset_path=tmpdir,
            config=train_config,
            num_threads=1,
            epochs=1,
            max_length=ml,
            train_val_ratio=0.2,
            device="cpu",
            mode=mode
        )
        assert os.path.exists(train_config["model_checkpoint"])
        assert isinstance(torch.load(train_config["model_checkpoint"]), tuple)


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="Setup does not support a cuda enabled graphics card")
def test_cuda_training(random_fasta, train_config, expected_labels):

    with TemporaryDirectory(prefix=PREFIX) as tmpdir:
        train_network(
            fasta=random_fasta,
            label_dir=expected_labels,
            dataset_path=tmpdir,
            config=train_config,
            num_threads=os.cpu_count(),
            epochs=1,
            max_length=20,
            train_val_ratio=0.2,
            device="cuda"
        )
        assert os.path.exists(train_config["model_checkpoint"])
        assert isinstance(torch.load(train_config["model_checkpoint"]), tuple)

