import os
import pickle
from tempfile import TemporaryDirectory
from typing import Dict, Union
import torch
from torch.utils.data import DataLoader
from Bio import SeqIO
import numpy as np
from RNAdist.NNModels.DISTAtteNCionE import (
    DISTAtteNCionE2, DISTAtteNCionESmall
)
from RNAdist.NNModels.Datasets import RNAPairDataset, RNAWindowDataset


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
    state_dict, config = torch.load(saved_model, map_location="cpu")
    if config["model"] == "normal":
        model = DISTAtteNCionE2(17, nr_updates=config["nr_layers"])
    elif config["model"] == "small":
        model = DISTAtteNCionESmall(17, nr_updates=config["nr_layers"])
    else:
        raise ValueError("Not able to infer model")
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
        step_size: int = 1,
        dataset_dir: str = None
):
    if dataset_dir is None:
        tmpdir = TemporaryDirectory()
        workdir = tmpdir.name
    else:
        workdir = dataset_dir
    dataset = RNAWindowDataset(
        data=fasta,
        label_dir=None,
        dataset_path=workdir,
        num_threads=num_threads,
        max_length=max_length,
        md_config=md_config,
        step_size=step_size
    )
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_threads,
        pin_memory=False,
    )
    model, config = _load_model(saved_model, device)
    current_data = {}
    output = {}
    sr_data = {sr.description: sr for sr in SeqIO.parse(fasta, "fasta")}
    for element in iter(data_loader):
        with torch.no_grad():

            x, pair_matrix, _, mask, indices = element
            pair_matrix = pair_matrix.to(device)
            mask = mask.to(device)
            if not config["masking"]:
                mask = None
            batched_pred = model(pair_matrix, mask=mask).cpu()
            batched_pred = batched_pred.numpy()
            for batch_index, index in enumerate(indices):

                _, idx, description = dataset.data_array[int(index)]
                pred = batched_pred[batch_index]
                if description not in current_data:
                    current_data[description] = {}
                current_data[description][idx] = pred
                if len(current_data) >= 2:
                    _handle_current_data(
                        current_data,
                        dataset,
                        sr_data,
                        max_length,
                        output
                    )
    _handle_current_data(current_data, dataset, sr_data, max_length, output)
    assert len(current_data) == 0
    if dataset_dir is None:
        tmpdir.cleanup()
    out_dir = os.path.dirname(outfile)
    if out_dir != "":
        os.makedirs(out_dir, exist_ok=True)
    with open(outfile, "wb") as handle:
        pickle.dump(output, handle)


def _handle_current_data(current_data, dataset, sr_data, max_length, output):
    keys = []
    for key, inner_dict in current_data.items():
        expected_indices = dataset.index_mapping[key]
        if len(expected_indices) == len(inner_dict):
            whole_pred = _puzzle_output(
                inner_dict,
                sr_data[key],
                max_length
            )
            output[key] = whole_pred
            keys.append(key)
    for key in keys:
        del current_data[key]


def _puzzle_output(predictions, seq_record, max_length):
    seq_len = len(seq_record.seq)
    whole_prediction = np.empty((seq_len, seq_len, len(predictions)))
    whole_prediction[:] = np.nan
    for i, (index, array) in enumerate(predictions.items()):
        end = index + max_length
        if end > seq_len:
            end = seq_len
            array_end = seq_len - index
            assert i == max(list(predictions.keys()))
            array = array[:array_end, :array_end]
        whole_prediction[index:end, index:end, i] = array
    whole_prediction = np.nanmean(whole_prediction, axis=-1)
    return whole_prediction


def _load_model(model_path, device):
    state_dict, config = torch.load(model_path, map_location="cpu")
    model = DISTAtteNCionE2(17, config["nr_layers"])
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, config


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