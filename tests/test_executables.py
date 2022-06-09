import os
import pytest
from RNAdist.Attention.training_set_generation import LabelDict
from RNAdist.Attention.tests.data_fixtures import (
    random_fasta,
    expected_labels,
    PREFIX,
    saved_model
)
import executables
from tempfile import TemporaryDirectory
import subprocess

EXECUTABLES_FILE = os.path.abspath(executables.__file__)


def test_cmd_training(random_fasta, expected_labels, tmp_path):
    model_file = os.path.join(tmp_path, "cmd_model.pt")
    with TemporaryDirectory(prefix=PREFIX) as tmpdir:

        process = ["python", EXECUTABLES_FILE,
                   "train",
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


def test_cmd_prediction(saved_model, random_fasta, tmp_path):
    prediction_out = os.path.join(tmp_path, "cmd_prediction.pckl")
    process = ["python", EXECUTABLES_FILE,
               "predict",
               "--input", random_fasta,
               "--output", prediction_out,
               "--model_file", saved_model,
               "--num_threads", str(os.cpu_count()),
               "--max_length", "20"
               ]
    data = subprocess.run(process, stderr=subprocess.PIPE)
    assert data.stderr.decode() == ""
    assert os.path.exists(prediction_out)


def test_cmd_data_generation(tmp_path, random_fasta):
    dataset = os.path.join(tmp_path, "test_dataset")
    process = ["python", EXECUTABLES_FILE,
               "generate_data",
               "--input", random_fasta,
               "--output", dataset,
               "--num_threads", str(os.cpu_count()),
               "--bin_size", "1",
               "--nr_samples", "1"
               ]
    data = subprocess.run(process, stderr=subprocess.PIPE)
    assert data.stderr.decode() == ""
    assert os.path.exists(dataset)
    ld = LabelDict(dataset)
