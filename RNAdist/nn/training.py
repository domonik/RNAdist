import os
from typing import Dict, List, Tuple, Callable, Any

import torch
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data import random_split
from Bio import SeqIO
from tempfile import TemporaryDirectory
import numpy as np
import pandas as pd
from RNAdist.nn.configuration import ModelConfiguration
from RNAdist.nn.DISTAtteNCionE import (
    RNADISTAtteNCionE,
    DISTAtteNCionESmall,
    WeightedDiagonalMSELoss
)
from RNAdist.nn.Datasets import RNAPairDataset, RNAWindowDataset, DataAugmentor, normalize_bpp, shift_index


def _loader_generation(
        training_set,
        validation_set,
        batch_size: int,
        num_threads: int = 1,
        sample: int = None
):
    if sample:
        train_loader = DataLoader(
            training_set,
            batch_size=batch_size,
            num_workers=num_threads,
            pin_memory=True,
            sampler=RandomSampler(
                data_source=training_set,
                replacement=True,
                num_samples=sample
            )
        )
        val_loader = DataLoader(
            validation_set,
            batch_size=batch_size,
            num_workers=num_threads,
            pin_memory=True,
            sampler=RandomSampler(
                data_source=validation_set,
                replacement=True,
                num_samples=sample
            )
        )
    else:
        train_loader = DataLoader(
            training_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_threads,
            pin_memory=True,
        )
        val_loader = DataLoader(
            validation_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_threads,
            pin_memory=True,
        )
    return train_loader, val_loader


def _split_fasta(fasta, train_val_ratio, label_dir, data_storage, num_threads, max_length, global_mask_size):
    with TemporaryDirectory(prefix="RNAdist_split") as tmpdir:
        seq_records = [seq_record for seq_record in SeqIO.parse(fasta, "fasta")]
        indices = torch.randperm(len(seq_records))
        t = int(len(indices) * train_val_ratio)
        train_seqs = [seq_records[idx] for idx in indices[:t]]
        val_seqs = [seq_records[idx] for idx in indices[t:]]
        train_file = os.path.join(tmpdir, "training_set")
        valid_file = os.path.join(tmpdir, "validation_set")
        with open(train_file, "w") as handle:
            SeqIO.write(train_seqs, handle, "fasta")
        with open(valid_file, "w") as handle:
            SeqIO.write(val_seqs, handle, "fasta")
        val_storage = os.path.join(data_storage, "validation")
        train_storage = os.path.join(data_storage, "training")
        train_set = RNAWindowDataset(
            data=train_file,
            label_dir=label_dir,
            dataset_path=train_storage,
            num_threads=num_threads,
            max_length=max_length,
            step_size=1,
            global_mask_size=global_mask_size

        )
        val_set = RNAWindowDataset(
            data=valid_file,
            label_dir=label_dir,
            dataset_path=val_storage,
            num_threads=num_threads,
            max_length=max_length,
            step_size=1,
            global_mask_size = global_mask_size

        )
        return train_set, val_set


def _dataset_generation(
        fasta: str,
        label_dir: str,
        data_storage: str,
        num_threads: int = 1,
        max_length: int = 200,
        train_val_ratio: float = 0.8,
        md_config: Dict = None,
        mode: str = "normal",
        global_mask_size: int = None,
        augmentor: DataAugmentor = None
):
    if mode == "normal":
        if global_mask_size\
                is not None:
            print("global_mask_size has no effect in normal mode")
        dataset = RNAPairDataset(
            data=fasta,
            label_dir=label_dir,
            dataset_path=data_storage,
            num_threads=num_threads,
            max_length=max_length,
            md_config=md_config,
            augmentor=augmentor
        )
        t = int(len(dataset) * train_val_ratio)
        v = len(dataset) - t
        training_set, validation_set = random_split(dataset, [t, v])
        validation_set.dataset.random_shift = None
    elif mode == "window":
        training_set, validation_set = _split_fasta(
            fasta=fasta,
            train_val_ratio=train_val_ratio,
            label_dir=label_dir,
            data_storage=data_storage,
            num_threads=num_threads,
            max_length=max_length,
            global_mask_size=global_mask_size

        )
    else:
        raise ValueError("Unsupported mode")
    print(f"training set size: {len(training_set)}")
    print(f"validation set size: {len(validation_set)}")
    return training_set, validation_set


def _setup(
        fasta: str,
        label_dir: str,
        data_storage: str,
        batch_size,
        num_threads: int = 1,
        max_length: int = 200,
        train_val_ratio: float = 0.2,
        md_config: Dict = None,
        seed: int = 0,
        mode: str = "normal",
        sample: int = None,
        global_mask_size: int = None,
        augmentor: DataAugmentor = None
):
    torch.manual_seed(seed)
    train_set, val_set = _dataset_generation(
        fasta=fasta,
        label_dir=label_dir,
        data_storage=data_storage,
        num_threads=num_threads,
        max_length=max_length,
        train_val_ratio=train_val_ratio,
        md_config=md_config,
        mode=mode,
        global_mask_size=global_mask_size,
        augmentor=augmentor
    )
    train_loader, val_loader = _loader_generation(
        train_set,
        val_set,
        batch_size=batch_size,
        num_threads=num_threads,
        sample=sample
    )

    return train_loader, val_loader


def _unpack_batch(batch, device, config):
    if config["masking"]:
        pair_rep, y, mask, _ = batch
        mask = mask.to(device)
        numel = torch.count_nonzero(mask)
    else:
        pair_rep, y, _, _ = batch
        numel = y.numel()
        mask = None
    y = y.to(device)
    pair_rep = pair_rep.to(device)
    indices = config.indices.to(device)
    pair_rep = torch.index_select(pair_rep, -1,  indices)
    return pair_rep, y, mask, numel


def _run_prediction(data_loader, model, losses, device, config, optimizer, train: bool = True):
    total_loss = 0
    total_mae = 0
    batch_idx = 0
    if train:
        model.train()
    else:
        model.eval()
    if isinstance(data_loader.dataset, RNAWindowDataset):
        ml = data_loader.dataset.max_length
        gms = data_loader.dataset.global_mask_size
        global_mask = torch.zeros(ml, ml, device=device)
        mid = int((ml-1) / 2)
        i_start = j_start = mid - gms
        i_end = j_end = mid + gms + 1
        global_mask[i_start:i_end, j_start:j_end] = 1
    else:
        global_mask = None
        i_start = i_end = None
    for batch_idx, batch in enumerate(iter(data_loader)):
        pair_rep, y, mask, numel = _unpack_batch(batch, device, config)
        pred = model(pair_rep, mask=mask)
        if global_mask is not None:
            pred = pred * global_mask
            y = y * global_mask
            numel = int((i_end - i_start) ** 2) * pred.shape[0]
        multi_loss = 0
        for criterion, weight, elementwise in losses:
            loss = criterion(y, pred)
            if not elementwise:
                loss = loss / numel  # adjusts the loss to be elementwise
            multi_loss = multi_loss + loss * weight
            multi_loss = multi_loss / config["gradient_accumulation"]
        if train:
            multi_loss.backward()
            if ((batch_idx + 1) % config["gradient_accumulation"] == 0) or (batch_idx + 1 == len(data_loader)):
                optimizer.step()
                optimizer.zero_grad()
        total_loss += multi_loss.item() * config["gradient_accumulation"]
        absolute_error = torch.sum(torch.abs(pred - y)) / numel
        total_mae += absolute_error.item()
    total_mae /= min((batch_idx + 1), len(data_loader.dataset) / config["batch_size"])
    total_loss /= min((batch_idx + 1), len(data_loader.dataset) / config["batch_size"])
    return total_loss, total_mae


def train_model(
        train_loader,
        val_loader,
        epochs: int,
        config: ModelConfiguration,
        device: str = None,
        seed: int = 0,
        fine_tune: str = None
):
    learning_rate = config["learning_rate"]
    patience = config["patience"]
    torch.manual_seed(seed)
    out_dir = os.path.dirname(config["model_checkpoint"])
    if config.training_stats is not None:
        if os.path.exists(config.training_stats):
            raise FileExistsError("Training stats file already exists. Please remove it or use another filename")
    header = ["epoch", "training loss", "training MAE", "validation loss", "validation MAE"]
    tstats = []
    if out_dir != "":
        os.makedirs(out_dir, exist_ok=True)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        assert device.startswith("cuda") or device.startswith("cpu")
    input_dim = config.input_dim

    if config["model"] == "normal":
        model = RNADISTAtteNCionE(
            input_dim,
            nr_updates=config["nr_layers"],
            checkpointing=config.gradient_checkpointing
        )
    elif config["model"] == "small":
        model = DISTAtteNCionESmall(
            input_dim,
            nr_updates=config["nr_layers"],
            checkpointing=config.gradient_checkpointing
        )
    elif isinstance(config["model"], torch.nn.Module):
        model = config["model"]
    else:
        raise ValueError("no valid Model Type")
    if fine_tune:
        state_dict, old_config = torch.load(fine_tune, map_location="cpu")
        if not isinstance(config["model"], torch.nn.Module):
            if old_config["model"] != config["model"]:
                raise ValueError("Model type of current configuration does not match the pretrained model type:\n"
                                 f"pretrained model: {old_config['model']}\n"
                                 f"current model: {config['model']}")
        else:
            if not isinstance(old_config["model"], type(config["model"])):
                raise ValueError("Model type of current configuration does not match the pretrained model type:\n"
                                 f"pretrained model: {type(old_config['model'])}\n"
                                 f"current model: {type(config['model'])}")
        model.load_state_dict(state_dict)
    model.to(device)
    opt = config["optimizer"].lower()
    if opt == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=config["momentum"],
            weight_decay=config["weight_decay"]
        )
    elif opt == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=config["weight_decay"]
        )
    else:
        raise ValueError("No valid optimizer provided")
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config["lr_step_size"] if isinstance(optimizer, torch.optim.SGD) else 1000,
        gamma=0.1 if isinstance(optimizer, torch.optim.SGD) else 1
        # prevents using scheduling if adaptive optimization is used
    )
    criterion = torch.nn.MSELoss(
        reduction="sum"
    )
    losses = [(criterion, 1, False)]
    best_epoch = 0
    best_val_loss = torch.tensor((float("inf")))
    best_val_mae = torch.tensor((float("inf")))
    epoch = 0
    for epoch in range(epochs):
        train_loss, train_mae = _run_prediction(
            train_loader, model, losses, device, config, optimizer, train=True
        )
        scheduler.step()
        if not epoch % config["validation_interval"]:
            with torch.no_grad():
                val_loss, val_mae = _run_prediction(
                    val_loader, model, losses, device, config, optimizer, train=False
                )
            if val_loss <= best_val_loss:
                best_val_loss = val_loss
                best_val_mae = val_mae
                best_epoch = epoch
                torch.save((model.state_dict(), config), config["model_checkpoint"])
            print(
                f"Epoch: {epoch}\tTraining Loss: {train_loss:.2f}\tTraining MAE: {train_mae:.2f}\tValidation Loss: {val_loss:.2f}\tValidation MAE: {val_mae:.2f}")
            tstats.append([epoch, train_loss, train_mae, val_loss, val_mae])
        else:
            print(f"Epoch: {epoch}\tTraining Loss: {train_loss:.2f}\tTraining MAE: {train_mae:.2f}")
            tstats.append([epoch, train_loss, train_mae, np.nan, np.nan])
        if config.training_stats is not None:
            df = pd.DataFrame(tstats, columns=header)
            df.to_csv(config.training_stats, sep="\t", index=False)
        if epoch - best_epoch >= patience:
            break
        if torch.isnan(torch.tensor(train_loss)):
            break
    best_val_mae = float(best_val_mae)
    return {"cost": best_val_mae, "epoch": epoch, "state_dict": model.state_dict()}


def train_network(fasta: str,
                  dataset_path: str,
                  label_dir: str,
                  config: ModelConfiguration,
                  num_threads: int = 1,
                  epochs: int = 400,
                  device: str = None,
                  max_length: int = 200,
                  train_val_ratio: float = 0.8,
                  md_config: Dict = None,
                  mode: str = "normal",
                  seed: int = 0,
                  fine_tune: str = None,
                  global_mask_size: int = None
                  ):
    """Python API for training a DISTAtteNCionE Network

    Args:
        fasta (str): Path to the Fasta file containing training sequences
        dataset_path (str): Path where the Dataset object will be stored 
        label_dir (str): Path to the directory created via
            :func:`~RNAdist.nn.training_set_generation.training_set_from_fasta`
        config (ModelConfiguration): configuration of training process
        num_threads (int): number of parallel processes to use
        epochs (int): maximum number of epochs
        device (str): one of cpu or cuda:x with x specifying the cuda device
        max_length (str): maximum length of the sequences used for padding or window generation
        train_val_ratio (float): part that is used for training. 1-train_val ratio is used for validation
        md_config (dict of str): !!Deprecated!! new versions will infer this from the label_dir
        mode (str): One of "normal" or "window". Specifies the mode that is used for training.
        seed (int): Random number seed for everything related to pytorch
        fine_tune (str): Path to a pretrained model that should be used for fine tuning.
        global_mask_size (int): Global Mask applied in window prediction mode. Has no effect in normal mode.

    Examples:
        You can train a network using the following lines  of code. The
        :class:`~RNAdist.nn.configuration.ModelConfiguration` object is mandatory but has only
        default values except for the path to the output file

        >>> from RNAdist.nn.training import train_network
        >>> from RNAdist.nn.configuration import ModelConfiguration
        >>> config = ModelConfiguration(model_checkpoint="path_to_output_model.pckl")
        >>> train_network("fasta.fa", "dataset_path", "label_directory", config=config)

        You can also change to window mode using a window size of 100 like this

        >>> train_network("fasta.fa", "dataset_path", "label_directory", config=config, mode="window", max_length=100)
    """
    aug_fcts = {"pair": [], "single": [], "single_prob": [], "pair_prob": []}
    if config.normalize_bpp:
        aug_fcts["pair"].append(normalize_bpp)
        aug_fcts["pair_prob"].append(1)
    if config.random_shift is not None:
        aug_fcts["single"].append(shift_index)
        aug_fcts["single_prob"].append(config.random_shift)
    if any((config.normalize_bpp, config.random_shift)):
        print("using Data Augmentation")
        augmentor = DataAugmentor(
            pair_rep_functions=aug_fcts["pair"],
            single_rep_functions=aug_fcts["single"],
            p_single=aug_fcts["single_prob"],
            p_pair=aug_fcts["pair_prob"]
        )
    else:
        augmentor = None
    train_loader, val_loader = _setup(
        fasta=fasta,
        label_dir=label_dir,
        data_storage=dataset_path,
        batch_size=config["batch_size"],
        num_threads=num_threads,
        max_length=max_length,
        train_val_ratio=train_val_ratio,
        md_config=md_config,
        seed=seed,
        mode=mode,
        sample=config.sample,
        global_mask_size=global_mask_size,
        augmentor=augmentor
    )
    train_return = train_model(
        train_loader,
        val_loader,
        epochs,
        config,
        device=device,
        seed=seed,
        fine_tune=fine_tune
    )
    return train_return["state_dict"]


def training_executable_wrapper(args):
    config = ModelConfiguration(
        masking=not args.no_masking,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        validation_interval=args.validation_interval,
        nr_layers=args.nr_layers,
        patience=args.patience,
        optimizer=args.optimizer,
        model_checkpoint=args.output,
        lr_step_size=args.learning_rate_step_size,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        model=args.model,
        gradient_accumulation=args.gradient_accumulation,
        sample=args.sample,
        use_bppm=not args.exclude_bppm,
        use_position=not args.exclude_position,
        normalize_bpp=args.normalize_bppm,
        random_shift=args.random_shift,
        gradient_checkpointing=args.gradient_checkpointing
    )

    train_network(
        fasta=args.input,
        label_dir=args.label_dir,
        dataset_path=args.dataset_path,
        num_threads=args.num_threads,
        max_length=args.max_length,
        config=config,
        seed=args.seed,
        epochs=args.max_epochs,
        device=args.device,
        fine_tune=args.fine_tune
    )


if __name__ == '__main__':
    pass
