import os
import pickle
from tempfile import TemporaryDirectory
from typing import Dict, Union
import torch
from torch.utils.data import DataLoader

from RNAdist.Attention.DISTAtteNCionE import (
    DISTAtteNCionE2, DISTAtteNCionESmall
)
from RNAdist.Attention.Datasets import RNAPairDataset


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
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_threads,
            pin_memory=False,
        )
        state_dict, config = torch.load(saved_model, map_location="cpu")
        if config["model"] == "normal":
            model = DISTAtteNCionE2(17, nr_updates=config["nr_layers"])
        elif config["model"] == "small":
            model = DISTAtteNCionESmall(17, nr_updates=config["nr_layers"])
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        output = {}
        for element in iter(data_loader):
            with torch.no_grad():
                x, bppm, y, mask, indices = element
                bppm = bppm.to(device)
                mask = mask.to(device)
                if not config["masking"]:
                    mask = None
                pred = model(bppm, mask=mask).cpu()
                pred = pred.numpy()
                for e_idx, idx in enumerate(indices):
                    description, seq = dataset.rna_graphs[idx]
                    e_pred = pred[e_idx][:len(seq), :len(seq)]
                    output[description] = e_pred
    out_dir = os.path.dirname(outfile)
    if out_dir != "":
        os.makedirs(out_dir, exist_ok=True)
    with open(outfile, "wb") as handle:
        pickle.dump(output, handle)


def prediction_executable_wrapper(args, md_config):
    model_predict(args.input,
                  args.model_file,
                  args.output,
                  batch_size=args.batch_size,
                  num_threads=args.num_threads,
                  device=args.device,
                  max_length=args.max_length,
                  md_config=md_config
                  )


if __name__ == '__main__':
    pass