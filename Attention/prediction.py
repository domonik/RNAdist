import argparse
import os
import pickle
from tempfile import TemporaryDirectory
from typing import Dict, Union
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader

from Attention.DISTAtteNCionE import DISTAtteNCionE2
from Attention.Datasets import RNAPairDataset


def model_predict(
        fasta: Union[str, os.PathLike],
        saved_model: Union[str, os.PathLike],
        outfile: Union[str, os.PathLike],
        batch_size: int = 1,  # TODO: Has to be fixed
        num_threads: int = 1,
        device: str = "cpu",
        max_length: int = 200,
        md_config: Dict = None):
    with TemporaryDirectory() as tmpdir:
        dataset = RNAPairDataset(
            data=fasta,
            label_dir=None,
            dataset_path=tmpdir,
            num_threads=num_threads,
            max_length=max_length,
            md_config=md_config
        )
        data_loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=1,
            pin_memory=False,
        )
        state_dict, config = torch.load(saved_model, map_location="cpu")
        model = DISTAtteNCionE2(17, config["nr_layers"])
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        output = {}
        for idx, element in enumerate(iter(data_loader)):
            with torch.no_grad():
                x, bppm, y, _ = element
                pred = model(bppm)
                pred = pred.numpy()
                description, _ = dataset.rna_graphs[idx]
                output[description] = pred
    with open(outfile, "wb") as handle:
        pickle.dump(output, handle)


def prediction_executable_wrapper(args, md_config):
    model_predict(args.input,
                  args.model_file,
                  args.output,
                  batch_size=1,
                  num_threads=1,
                  device="cpu",
                  max_length=200,
                  md_config=md_config
                  )



if __name__ == '__main__':
    pass