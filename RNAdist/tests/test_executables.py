import os
from RNAdist.NNModels.training_set_generation import LabelDict
from RNAdist.NNModels.tests.data_fixtures import (
    PREFIX
)
from RNAdist import distattencione_executables, executables
from tempfile import TemporaryDirectory
import subprocess
import pickle
from Bio import SeqIO
import pytest
import RNAdist

DISTATT_EXECUTABLES_FILE = os.path.abspath(distattencione_executables.__file__)
EXECUTABLES_FILE = os.path.abspath(executables.__file__)
env = os.environ.copy()
env["PYTHONPATH"] = os.path.abspath(os.path.dirname(os.path.dirname(RNAdist.__file__)))


def test_cmd_training(random_fasta, expected_labels, tmp_path):
    model_file = os.path.join(tmp_path, "cmd_model.pt")
    with TemporaryDirectory(prefix=PREFIX) as tmpdir:
        process = ["python", DISTATT_EXECUTABLES_FILE,
                   "train",
                   "--input", random_fasta,
                   "--label_dir", expected_labels,
                   "--output", model_file,
                   "--data_path", tmpdir,
                   "--max_length", "20",
                   "--max_epochs", "1"
                   ]
        data = subprocess.run(process, stderr=subprocess.PIPE, env=env)
        assert data.stderr.decode() == ""
        assert os.path.exists(model_file)


def test_cmd_prediction(saved_model, random_fasta, tmp_path):
    desc = set(sr.description for sr in SeqIO.parse(random_fasta, "fasta"))
    prediction_out = os.path.join(tmp_path, "cmd_prediction.pckl")
    process = ["python", DISTATT_EXECUTABLES_FILE,
               "predict",
               "--input", random_fasta,
               "--output", prediction_out,
               "--batch_size", "4",
               "--model_file", saved_model,
               "--num_threads", str(os.cpu_count()),
               "--max_length", "20"
               ]
    data = subprocess.run(process, stderr=subprocess.PIPE, env=env)
    assert data.stderr.decode() == ""
    assert os.path.exists(prediction_out)
    with open(prediction_out, "rb") as handle:
        data = pickle.load(handle)
    for key in desc:
        assert key in data


def test_cmd_data_generation(tmp_path, random_fasta):
    dataset = os.path.join(tmp_path, "test_dataset")
    process = ["python", DISTATT_EXECUTABLES_FILE,
               "generate_data",
               "--input", random_fasta,
               "--output", dataset,
               "--num_threads", str(os.cpu_count()),
               "--bin_size", "1",
               "--nr_samples", "1"
               ]
    data = subprocess.run(process, stderr=subprocess.PIPE, env=env)
    assert data.stderr.decode() == ""
    assert os.path.exists(dataset)
    ld = LabelDict(dataset)


@pytest.mark.parametrize(
    "command",
    [
        "sample",
        "clote-ponty",
        "pmcomp"
    ]
)
def test_rnadist_cmd(tmp_path, random_fasta, command):
    op = os.path.join(tmp_path, "test_data.pckl")
    process = [
        "python", EXECUTABLES_FILE, command,
        "--input", random_fasta,
        "--output", op,
        "--num_threads", str(os.cpu_count()),
    ]
    data = subprocess.run(process, stderr=subprocess.PIPE, env=env)
    assert data.stderr.decode() == ""
    assert os.path.exists(op)
    with open(op, "rb") as handle:
        data = pickle.load(handle)
    for sr in SeqIO.parse(random_fasta, "fasta"):
        assert sr.description in data