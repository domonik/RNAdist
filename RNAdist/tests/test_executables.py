import os
from RNAdist.nn.training_set_generation import LabelDict
from RNAdist import distattencione_executables, executables
from tempfile import TemporaryDirectory
import subprocess
import pickle
from Bio import SeqIO
import pytest
import sys
import RNAdist
import pandas as pd

pytest_plugins = [
    "RNAdist.dp.tests.fixtures",
    "RNAdist.nn.tests.data_fixtures",
    "RNAdist.tests.fasta_fixtures"
]

DISTATT_EXECUTABLES_FILE = os.path.abspath(distattencione_executables.__file__)
EXECUTABLES_FILE = os.path.abspath(executables.__file__)
env = os.environ.copy()
env["PYTHONPATH"] = ":".join(([os.path.abspath(os.path.dirname(os.path.dirname(RNAdist.__file__)))] + sys.path))


def test_cmd_training(random_fasta, expected_labels, tmp_path, prefix):
    model_file = os.path.join(tmp_path, "cmd_model.pt")
    with TemporaryDirectory(prefix=prefix) as tmpdir:
        process = ["python", DISTATT_EXECUTABLES_FILE,
                   "train",
                   "--input", random_fasta,
                   "--label_dir", expected_labels,
                   "--output", model_file,
                   "--dataset_path", tmpdir,
                   "--max_length", "20",
                   "--max_epochs", "1"
                   ]
        data = subprocess.run(process, stderr=subprocess.PIPE, env=env)
        assert data.stderr.decode() == ""
        assert os.path.exists(model_file)


def test_cmd_hpo(random_fasta, expected_labels, tmpdir):
    model = os.path.join(tmpdir, "smac_model.pt")
    smac_dir = os.path.join(tmpdir, "smac_dir")
    dataset_path = os.path.join(tmpdir, "smac_dataset")
    process = ["python", DISTATT_EXECUTABLES_FILE,
               "optimize",
               "--input", random_fasta,
               "--label_dir", expected_labels,
               "--output", model,
               "--dataset_path", dataset_path,
               "--run_default",
               "--max_length", "20",
               "--max_epochs", "1",
               "--ta_run_limit", "5",
               "--device", "cpu",
               "--smac_dir", smac_dir
               ]
    _ = subprocess.run(process, stderr=subprocess.PIPE, env=env)
    assert os.path.exists(model)


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
        "clote-ponty",
        "pmcomp",
        "sample",
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

@pytest.mark.parametrize(
    "names", ["bed_test", None]
)
def test_binding_site_executable(tmp_path, bed_test_fasta, bed_test_bed, names):
    op = os.path.join(tmp_path, "test_data.pckl")
    process = [
        "python", EXECUTABLES_FILE, "binding-site",
        "--input", bed_test_fasta,
        "--bed_files", bed_test_bed, bed_test_bed,
        "--output", op,
        "--num_threads", str(os.cpu_count()),
    ]
    if names is not None:
        process += ["--names", "bed1", "bed2"]
    data = subprocess.run(process, stderr=subprocess.PIPE, env=env)
    assert data.stderr.decode() == ""
    assert os.path.exists(op)
    df = pd.read_csv(op, sep="\t")
    assert df.shape[0] >= 1
    assert df.shape[1] == 8


def test_rnadist_extract(tmp_path, example_output, example_output_path):
    process = [
        "python", EXECUTABLES_FILE, "extract",
        "--data_file", example_output_path,
        "--outdir", tmp_path,
    ]
    data = subprocess.run(process, stderr=subprocess.PIPE, env=env)
    assert data.stderr.decode() == ""
    for key in example_output.keys():
        expected_path = os.path.join(tmp_path, f"{key}.tsv")
        assert os.path.exists(expected_path)
        df = pd.read_csv(expected_path, sep="\t")
        assert len(df != 0)
