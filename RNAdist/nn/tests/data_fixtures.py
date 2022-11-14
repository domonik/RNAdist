import os
import pytest
import torch
import math
from RNAdist.nn.configuration import ModelConfiguration

TESTFILE_DIR = os.path.dirname(os.path.abspath(__file__))
TESTDATA_DIR = os.path.join(TESTFILE_DIR, "test_data")


@pytest.fixture()
def prefix():
    return "RNAdist_"


@pytest.fixture()
def generation_config():
    config = {
        "temperature": 37,
        "min_loop_size": 3,
        "noGU": 0,
    }
    return config


@pytest.fixture()
def window_config():
    config = {
        "temperature": 37,
        "min_loop_size": 3,
        "noGU": 0,
        "max_bp_span": 10,
        "window_size":10
    }
    return config


@pytest.fixture
def expected_labels():
    return os.path.join(TESTDATA_DIR, "expected_labels")

@pytest.fixture
def expected_window_labels():
    return os.path.join(TESTDATA_DIR, "expected_window_labels")


@pytest.fixture
def random_fasta():
    return os.path.join(TESTDATA_DIR, "random_test.fa")


@pytest.fixture
def train_config(tmp_path):
    config = ModelConfiguration(
        masking=True,
        learning_rate=0.01,
        batch_size=4,
        validation_interval=5,
        nr_layers=1,
        patience=20,
        optimizer="adamw",
        model_checkpoint=os.path.join(tmp_path, "test_model.pt"),
        lr_step_size=1,
        weight_decay=0,
        model="normal",
        gradient_accumulation=2,
    )
    return config



@pytest.fixture()
def saved_model():
    return os.path.join(TESTDATA_DIR, "test_model.pt")


@pytest.fixture()
def saved_model_no_bpp():
    return os.path.join(TESTDATA_DIR, "test_model_no_bpp.pt")


@pytest.fixture()
def expected_rna_data():
    data = torch.load(os.path.join(TESTDATA_DIR, "rna_tensor.pt"))
    return data


@pytest.fixture(scope="session")
def masked_pair_rep_batch():
    torch.manual_seed(42)
    pair_rep = torch.randn(3, 5, 5, 2)
    masks = []
    for _ in range(pair_rep.shape[0]):
        mask = torch.zeros(pair_rep.shape[1], pair_rep.shape[2])
        len = int(torch.randint(2, 4, (1,)))
        mask[0:len, 0:len] = 1
        masks.append(mask)
    masks = torch.stack(masks)
    pair_rep = pair_rep * masks[..., None]
    target = torch.ones(pair_rep.shape)
    assert masks.shape == pair_rep.shape[:-1]
    return pair_rep, masks, target
