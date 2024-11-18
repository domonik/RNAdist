import os
from RNAdist import executables
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
    "RNAdist.tests.fasta_fixtures"
]

EXECUTABLES_FILE = os.path.abspath(executables.__file__)
env = os.environ.copy()
env["PYTHONPATH"] = ":".join(([os.path.abspath(os.path.dirname(os.path.dirname(RNAdist.__file__)))] + sys.path))







@pytest.mark.parametrize(
    "command,redundant",
    [
        ("clote-ponty", False),
        ("pmcomp", False),
        ("sample", False),
        ("sample", True),
    ]
)
def test_rnadist_cmd(tmp_path, random_fasta, command, redundant):
    op = os.path.join(tmp_path, "test_data.pckl")
    process = [
        "python", EXECUTABLES_FILE, command,
        "--input", random_fasta,
        "--output", op,
        "--num_threads", str(os.cpu_count()),
    ]
    if command == "sample" and redundant:
        process += ["--non_redundant", "--nr_samples", "5"]
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
