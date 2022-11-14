from RNAdist.nn.prediction import model_predict, model_window_predict
from RNAdist.nn.tests.data_fixtures import (
    saved_model,
    saved_model_no_bpp,
    random_fasta
)
import os
import pickle
from Bio import SeqIO
import pytest
import numpy as np
import torch

pytest_plugins = ["RNAdist.dp.tests.fixtures",
                  "RNAdist.nn.tests.data_fixtures"]

@pytest.mark.parametrize(
    "model",
    [
        "saved_model",
        "saved_model_no_bpp"
    ]
)
def test_model_predict(model, random_fasta, tmp_path, request):
    saved_model = request.getfixturevalue(model)
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
        max_length=11,
        device="cuda",
        global_mask_size=1
    )
    assert os.path.exists(outfile)
    with open(outfile, "rb") as handle:
        data = pickle.load(handle)
    for key in desc:
        assert key in data


@pytest.mark.parametrize(
    "model", ["saved_model", "saved_model_no_bpp"]
)
@pytest.mark.parametrize("step_size", [1, 3])
@pytest.mark.parametrize("global_mask_size", [None, 3])
def test_window_predict(model, random_fasta, tmp_path, request, step_size, global_mask_size):
    saved_model = request.getfixturevalue(model)
    desc = [sr for sr in SeqIO.parse(random_fasta, "fasta")]
    outfile = os.path.join(tmp_path, "predictions")
    ml = 11
    model_window_predict(
        fasta=random_fasta,
        outfile=outfile,
        saved_model=saved_model,
        batch_size=4,
        num_threads=os.cpu_count(),
        max_length=ml,
        device="cpu",
        global_mask_size=global_mask_size,
        step_size=step_size
    )
    assert os.path.exists(outfile)
    gms = global_mask_size if global_mask_size is not None else int((ml - 1) / 2)
    with open(outfile, "rb") as handle:
        data = pickle.load(handle)
    for seq_record in desc:
        assert seq_record.description in data
        pred = data[seq_record.description]
        assert not np.all(pred[0, 0:1+gms] == 0)
        assert pred.shape[0] == len(seq_record.seq)
