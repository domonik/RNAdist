from RNAdist.Attention.prediction import model_predict, model_window_predict
from RNAdist.Attention.tests.data_fixtures import (
    saved_model,
    random_fasta
)
import os
import pickle
from Bio import SeqIO
import pytest
import torch


def test_model_predict(saved_model, random_fasta, tmp_path):
    desc = set(sr.description for sr in SeqIO.parse(random_fasta, "fasta"))
    outfile = os.path.join(tmp_path, "predictions")
    model_predict(
        fasta=random_fasta,
        outfile=outfile,
        saved_model=saved_model,
        batch_size=4,
        num_threads=os.cpu_count(),
        max_length=20
    )
    assert os.path.exists(outfile)
    with open(outfile, "rb") as handle:
        data = pickle.load(handle)
    for key in desc:
        assert key in data


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="Setup does not support a cuda enabled graphics card")
def test_cuda_predict(saved_model, random_fasta, tmp_path):
    desc = set(sr.description for sr in SeqIO.parse(random_fasta, "fasta"))
    outfile = os.path.join(tmp_path, "predictions")
    model_predict(
        fasta=random_fasta,
        outfile=outfile,
        saved_model=saved_model,
        batch_size=4,
        num_threads=os.cpu_count(),
        max_length=20,
        device="cuda"
    )
    assert os.path.exists(outfile)
    with open(outfile, "rb") as handle:
        data = pickle.load(handle)
    for key in desc:
        assert key in data


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="Setup does not support a cuda enabled graphics card")
def test_cuda_window_predict(saved_model, random_fasta, tmp_path):
    desc = set(sr.description for sr in SeqIO.parse(random_fasta, "fasta"))
    outfile = os.path.join(tmp_path, "predictions")
    model_window_predict(
        fasta=random_fasta,
        outfile=outfile,
        saved_model=saved_model,
        batch_size=4,
        num_threads=os.cpu_count(),
        max_length=10,
        device="cuda",
        step_size=1
    )
    assert os.path.exists(outfile)
    with open(outfile, "rb") as handle:
        data = pickle.load(handle)
    for key in desc:
        assert key in data


def test_window_predict(saved_model, random_fasta, tmp_path):
    desc = set(sr.description for sr in SeqIO.parse(random_fasta, "fasta"))
    outfile = os.path.join(tmp_path, "predictions")
    model_window_predict(
        fasta=random_fasta,
        outfile=outfile,
        saved_model=saved_model,
        batch_size=4,
        num_threads=os.cpu_count(),
        max_length=10,
        device="cpu",
        step_size=1
    )
    assert os.path.exists(outfile)
    with open(outfile, "rb") as handle:
        data = pickle.load(handle)
    for key in desc:
        assert key in data
