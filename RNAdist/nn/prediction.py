import os
import pickle
from tempfile import TemporaryDirectory
from typing import Dict, Union
import torch
from torch_geometric.loader import DataLoader
from Bio import SeqIO
import numpy as np

import RNAdist.nn.configuration
from RNAdist.nn.DISTAtteNCionE import (
    RNADISTAtteNCionE, DISTAtteNCionESmall, GraphRNADISTAtteNCionE
)
from RNAdist.nn.Datasets import RNAPairDataset, RNAWindowDataset, RNAGeometricInferenceDataset
from RNAdist.fasta_wrappers import md_config_from_args

@torch.no_grad()
def model_predict(
        fasta: Union[str, os.PathLike],
        saved_model: Union[str, os.PathLike],
        outfile: Union[str, os.PathLike],
        batch_size: int = 1,
        num_threads: int = 1,
        device: str = "cpu",
        max_length: int = 200,
        md_config: Dict = None,
        dataset_dir: str = None
):
    """Python API for predicting expected distances via a DISTAtteNCionE Network

    Args:
        fasta (str, os.PathLike): Path to fasta file containing sequences to predict
        saved_model (str, os.PathLike): Path to the model file generated via training of HPO
        outfile (str, os.PathLike): Path where to store the pickled expected distance predictions.
            File will contain a dictionary using the :code:`fasta` headers as key and the prediction as numpy
            array as value
        batch_size (int): batch_size used for prediction
        num_threads (int): number of parallel processes to use
        device (str): One of :code:`cpu` or :code:`cuda:x` with x specifying the cuda device
        max_length (int): Maximum length of the sequences used for padding.
        md_config (Dict): configuration dict used to change
            ViennaRNA model details
        dataset_dir: Path where to store the Dataset. If None a temporary directory will be created.
    """
    if dataset_dir is None:
        tmpdir = TemporaryDirectory()
        workdir = tmpdir.name
    else:
        workdir = dataset_dir
    dataset = RNAPairDataset(
        data=fasta,
        label_dir=None,
        dataset_path=workdir,
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
    model, config = _load_model(saved_model, device)
    model.eval()
    output = {}
    use = config.indices.to(device)
    for batch in iter(data_loader):
            batch = batch.to(device)
            pair_rep = batch["pair_rep"]
            pair_rep = torch.index_select(pair_rep, -1, use)
            batch["pair_rep"] = pair_rep
            mask = batch["mask"]
            if not config["masking"]:
                mask = None
            pred = model(batch, mask=mask).cpu()
            pred = pred.numpy()
            for e_idx, idx in enumerate(batch["item"]):
                description, seq = dataset.rna_graphs[idx]
                e_pred = pred[e_idx][:len(seq), :len(seq)]
                output[description] = e_pred
    out_dir = os.path.dirname(outfile)
    if out_dir != "":
        os.makedirs(out_dir, exist_ok=True)
    with open(outfile, "wb") as handle:
        pickle.dump(output, handle)
    if dataset_dir is None:
        tmpdir.cleanup()


def model_window_predict(
        fasta: Union[str, os.PathLike],
        saved_model: Union[str, os.PathLike],
        outfile: Union[str, os.PathLike],
        batch_size: int = 1,
        num_threads: int = 1,
        device: str = "cpu",
        max_length: int = 200,
        md_config: Dict = None,
        global_mask_size: int = None,
        dataset_dir: str = None,
        step_size: int = 1,
):
    if dataset_dir is None:
        tmpdir = TemporaryDirectory()
        workdir = tmpdir.name
    else:
        tmpdir = None
        workdir = dataset_dir
    dataset = RNAWindowDataset(
        data=fasta,
        label_dir=None,
        dataset_path=workdir,
        num_threads=num_threads,
        max_length=max_length,
        md_config=md_config,
        step_size=step_size,
        global_mask_size=global_mask_size
    )
    ml = inf_batch_size = None

    model, config = _load_model(saved_model, device, ml, inf_batch_size)
    model.eval()

    config: RNAdist.nn.configuration.ModelConfiguration



    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_threads,
        pin_memory=True if "cuda" in device else False,
        pin_memory_device=device,
    )
    output_data = {
        # this pads the sequence just like the dataset does in __getitem__
        desc: np.zeros((len(seq) + dataset.max_length - 1, len(seq) + dataset.max_length - 1))
        for desc, seq in dataset.seq_data
    }
    gms = global_mask_size if global_mask_size is not None else int((max_length - 1) / 2)
    mid = int((max_length-1) / 2)
    pred_s, pred_e = mid - gms, mid + gms + 1
    use = config.indices.to(device)

    for batch in iter(data_loader):
        with torch.no_grad():
            batch = batch.to(device)
            pair_rep = batch["pair_rep"]
            pair_rep = torch.index_select(pair_rep, -1, use)
            batch["pair_rep"] = pair_rep
            mask = batch["mask"]
            index_data = batch["idx_info"]
            if not config["masking"]:
                mask = None
            batched_pred = model(batch, mask=mask).cpu()
            batched_pred = batched_pred.numpy()
            for batch_index, _ in enumerate(index_data):
                file_idx = index_data[batch_index][0]
                i, j = index_data[batch_index][1:] + mid
                description, _ = dataset.seq_data[file_idx]
                pred = batched_pred[batch_index]
                output_data[description][i-gms:i+gms+1, j-gms:j+gms+1] = pred[pred_s:pred_e, pred_s:pred_e]
    for key, value in output_data.items():
        output_data[key] = value[mid:-mid, mid:-mid]
    if tmpdir is not None:
        tmpdir.cleanup()
    out_dir = os.path.dirname(outfile)
    if out_dir != "":
        os.makedirs(out_dir, exist_ok=True)
    with open(outfile, "wb") as handle:
        pickle.dump(output_data, handle)


def graph_predict(
        fasta: Union[str, os.PathLike],
        saved_model: Union[str, os.PathLike],
        outfile: Union[str, os.PathLike],
        batch_size: int = 1,
        num_threads: int = 1,
        device: str = "cpu",
        max_length: int = 200,
        md_config: Dict = None,
        dataset_dir: str = None,
):
    if dataset_dir is None:
        tmpdir = TemporaryDirectory()
        workdir = tmpdir.name
    else:
        tmpdir = None
        workdir = dataset_dir
    dataset = RNAGeometricInferenceDataset(
        data=fasta,
        label_dir=None,
        dataset_path=workdir,
        num_threads=num_threads,
        max_length=max_length,
        md_config=md_config,
    )
    ml = dataset.max_length
    inf_batch_size = batch_size
    model, config = _load_model(saved_model, device, ml, inf_batch_size)
    use = config.indices.to(device)

    model.eval()
    data_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_threads,
        pin_memory=True if "cuda" in device else False,
        pin_memory_device=device,
    )
    output_data = {}
    for batch_idx, batch in enumerate(iter(data_loader)):
        with torch.no_grad():
            batch = batch.to(device)
            pair_rep = batch["pair_rep"]
            pair_rep = torch.index_select(pair_rep, -1, use)
            batch["pair_rep"] = pair_rep
            mask = batch["mask"]
            if not config["masking"]:
                mask = None
            pred = model(batch, mask=mask).cpu()
            pred = pred.numpy()
            description, _ = dataset.seq_data[batch_idx]
            output_data[description] = pred
    if tmpdir is not None:
        tmpdir.cleanup()
    out_dir = os.path.dirname(outfile)
    if out_dir != "":
        os.makedirs(out_dir, exist_ok=True)
    with open(outfile, "wb") as handle:
        pickle.dump(output_data, handle)






def _load_model(model_path, device, ml: int = None, batch_size: int = None):
    state_dict, config = torch.load(model_path, map_location="cpu")
    if config["model"] == "normal":
        model = RNADISTAtteNCionE(config.input_dim, nr_updates=config["nr_layers"])
    elif config["model"] == "small":
        model = DISTAtteNCionESmall(config.input_dim, nr_updates=config["nr_layers"])
    elif config.model == "graph":
        config.gradient_checkpointing = False
        model = GraphRNADISTAtteNCionE(
            input_dim=9,
            embedding_dim=32,
            pair_dim=config.input_dim,
            max_length=ml,
            upper_bound=1,
            checkpointing=config.gradient_checkpointing,
            graph_layers=config.nr_layers,
            device=device,
            inference=True,
            inference_batch_size=batch_size
        )
    elif isinstance(config["model"], torch.nn.Module):
        model = config["model"]
    else:
        raise ValueError("Not able to infer model")
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, config


def prediction_executable_wrapper(args):
    md_config = md_config_from_args(args)
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
    from tempfile import TemporaryDirectory
    saved_model = "/home/rabsch/PythonProjects/RNAdist/RNAdist/nn/tests/test_data/test_model.pt"
    random_fasta = "/home/rabsch/PythonProjects/RNAdist/RNAdist/nn/tests/test_data/random_test.fa"
    desc = set(sr.description for sr in SeqIO.parse(random_fasta, "fasta"))
    with TemporaryDirectory() as tmp_path:
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