import os
import pytest
import torch

TESTFILE_DIR = os.path.dirname(os.path.abspath(__file__))
TESTDATA_DIR = os.path.join(TESTFILE_DIR, "test_data")
PREFIX = "RNAdist_"



@pytest.fixture()
def generation_config():
    config = {
        "temperature": 37,
        "min_loop_size": 3,
        "noGU": 0,
    }
    return config


@pytest.fixture
def expected_labels():
    return os.path.join(TESTDATA_DIR, "expected_labels")


@pytest.fixture
def random_fasta():
    return os.path.join(TESTDATA_DIR, "random_test.fa")


@pytest.fixture
def train_config(tmp_path):
    config = {
        "alpha": 0.99,
        "masking": True,
        "learning_rate": 0.01,
        "batch_size": 4,
        "validation_interval": 5,
        "nr_layers": 1,
        "patience": 20,
        "optimizer": "adamw",
        "model_checkpoint": os.path.join(tmp_path, "test_model.pt"),
        "lr_step_size": 1,
        "weight_decay": 0,
        "model": "normal",
        "gradient_accumulation": 2
    }
    return config


@pytest.fixture()
def saved_model():
    return os.path.join(TESTDATA_DIR, "test_model.pt")


@pytest.fixture()
def expected_rna_data():
    data = torch.load(os.path.join(TESTDATA_DIR, "rna_tensor.pt"))
    return data