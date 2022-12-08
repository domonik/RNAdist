from RNAdist.nn.training import train_network
from tempfile import TemporaryDirectory
import os
import torch
import pytest
import pandas as pd

pytest_plugins = ["RNAdist.dp.tests.fixtures",
                  "RNAdist.nn.tests.data_fixtures"]


class Modeltotest(torch.nn.Module):
    def __init__(self, embedding_size):
        super().__init__()
        self.embedding_size = embedding_size
        self.output = torch.nn.Linear(self.embedding_size, 1)

    def forward(self, pair_rep, mask):
        out = self.output(pair_rep)
        out = torch.squeeze(out)
        out = torch.relu(out)
        if mask is not None:
            out = out * mask
        return out


def triangularselfattention(dim):
    model = Modeltotest(dim)
    return model


@pytest.mark.parametrize("model_type", ["normal", "small", triangularselfattention])
@pytest.mark.parametrize("use_bppm", [True, False])
@pytest.mark.parametrize("use_pos", [True, False])
@pytest.mark.parametrize("mode", ["normal", "window"])
@pytest.mark.parametrize("random_shift", [None, 0.8])
@pytest.mark.parametrize("gradient_cp,nr_updates", [(True, 2), (False, 1)])
def test_training(random_fasta, train_config, expected_labels,
                  model_type, use_bppm, use_pos, expected_window_labels, mode, random_shift, prefix, gradient_cp, nr_updates):
    if random_shift is not None:
        train_config.random_shift = random_shift
    train_config.gradient_checkpointing = gradient_cp
    train_config.use_bppm = use_bppm
    train_config.use_position = use_pos
    if model_type not in ["small", "normal"]:
        if gradient_cp:  # Not necessary to check for that configuration
            return
        model_type = model_type(train_config.input_dim)
    train_config.model = model_type
    train_config.nr_layers = nr_updates
    if mode == "normal":
        expected_labels = expected_labels
        ml = 20
    else:
        expected_labels = expected_window_labels
        ml = 7
        train_config.sample = 10
    with TemporaryDirectory(prefix=prefix) as tmpdir:
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
            mode=mode,
        )
        assert os.path.exists(train_config["model_checkpoint"])
        state_dict, config = torch.load(train_config["model_checkpoint"])
        if not use_bppm:
            p=0
        assert isinstance(torch.load(train_config["model_checkpoint"]), tuple)
        assert state_dict["output.weight"].shape[-1] == len(config.indices)


def test_training_stats(random_fasta, expected_labels, tmpdir, train_config):
    train_config.training_stats = os.path.join(tmpdir, "training_stats.tsv")
    dataset_path = os.path.join(tmpdir, "dataset")
    epochs = 1
    train_network(
        fasta=random_fasta,
        label_dir=expected_labels,
        dataset_path=dataset_path,
        config=train_config,
        num_threads=1,
        epochs=epochs,
        max_length=20,
        train_val_ratio=0.2,
        device="cpu",
        mode="normal"
    )
    assert os.path.exists(train_config.training_stats)
    df = pd.read_csv(train_config.training_stats, sep="\t")
    assert df.shape == (1, 5)


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="Setup does not support a cuda enabled graphics card")
def test_cuda_training(random_fasta, train_config, expected_labels, prefix):

    with TemporaryDirectory(prefix=prefix) as tmpdir:
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


@pytest.mark.parametrize(
    "epochs",
    [0, 1]
)
def test_pretrained(random_fasta, train_config, expected_labels, saved_model, expected_window_labels, epochs, prefix):
    with TemporaryDirectory(prefix=prefix) as tmpdir:
        model_state_dict = train_network(
            fasta=random_fasta,
            label_dir=expected_labels,
            dataset_path=tmpdir,
            config=train_config,
            num_threads=1,
            epochs=epochs,
            max_length=20,
            train_val_ratio=0.2,
            device="cpu",
            fine_tune=saved_model
        )
        expected_state_dict, config = torch.load(saved_model)
        for key in expected_state_dict:
            expected_tensor = expected_state_dict[key]
            assert key in model_state_dict, "Keys of expected and actual model do not match"
            actual_tensor = model_state_dict[key]
            if epochs == 0:
                # Since we are training for 0 epochs the state dict should not be updated
                assert torch.equal(expected_tensor, actual_tensor), "Model loading in fine tuning failed"