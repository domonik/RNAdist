import os
from typing import Dict, List, Tuple, Callable, Any

import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from RNAdist.NNModels.DISTAtteNCionE import (
    DISTAtteNCionE2,
    DISTAtteNCionESmall,
    WeightedDiagonalMSELoss
)
from RNAdist.NNModels.Datasets import RNAPairDataset, RNAWindowDataset


def _loader_generation(
        training_set,
        validation_set,
        batch_size: int,
        num_threads: int = 1
):
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


def _dataset_generation(
        fasta: str,
        label_dir: str,
        data_storage: str,
        num_threads: int = 1,
        max_length: int = 200,
        train_val_ratio: float = 0.8,
        md_config: Dict = None,
        mode: str = "normal"
):
    if mode == "normal":
        dataset = RNAPairDataset(
            data=fasta,
            label_dir=label_dir,
            dataset_path=data_storage,
            num_threads=num_threads,
            max_length=max_length,
            md_config=md_config
        )
    elif mode == "window":
        dataset = RNAWindowDataset(
            data=fasta,
            label_dir=label_dir,
            dataset_path=data_storage,
            num_threads=num_threads,
            max_length=max_length,
            step_size=1
        )
    else:
        raise ValueError("Unsupported mode")
    t = int(len(dataset) * train_val_ratio)
    v = len(dataset) - t
    training_set, validation_set = random_split(dataset, [t, v])
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
        mode: str = "normal"
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
        mode=mode
    )
    train_loader, val_loader = _loader_generation(
        train_set,
        val_set,
        batch_size=batch_size,
        num_threads=num_threads
    )

    return train_loader, val_loader


def _unpack_batch(batch, device, config):
    if config["masking"]:
        _, pair_rep, y, mask, _ = batch
        mask = mask.to(device)
        numel = torch.count_nonzero(mask)
    else:
        _, pair_rep, y, _, _ = batch
        numel = y.numel()
        mask = None
    y = y.to(device)
    pair_rep = pair_rep.to(device)
    return pair_rep, y, mask, numel


def _train(model, data_loader, optimizer, device,
           losses: List[Tuple[Callable, float]], config: Dict):
    total_loss = 0
    model.train()
    optimizer.zero_grad()
    batch_idx = 0
    for batch_idx, batch in enumerate(iter(data_loader)):
        pair_rep, y, mask, numel = _unpack_batch(batch, device, config)
        pred = model(pair_rep, mask=mask)
        multi_loss = 0
        for criterion, weight in losses:
            loss = criterion(y, pred, mask)
            multi_loss = multi_loss + loss * weight
            multi_loss = multi_loss / config["gradient_accumulation"]
        multi_loss.backward()
        if ((batch_idx + 1) % config["gradient_accumulation"] == 0) or (
                batch_idx + 1 == len(data_loader)):
            optimizer.step()
            optimizer.zero_grad()
        total_loss += multi_loss.item() * y.shape[0] * config["gradient_accumulation"]
        if batch_idx >= config["sample"]:
            break
    total_loss /= batch_idx + 1
    return total_loss


def _validate(model, data_loader, device, losses: List[Tuple[Callable, float]],
              config, train_val_ratio):
    total_loss = 0
    total_mae = 0
    model.eval()
    idx = 0
    with torch.no_grad():
        for idx, batch in enumerate(iter(data_loader)):
            pair_rep, y, mask, numel = _unpack_batch(batch, device, config)
            pred = model(pair_rep, mask=mask)
            multi_loss = 0
            for criterion, weight in losses:
                loss = criterion(y, pred, mask)
                multi_loss = multi_loss + loss * weight
            total_loss += multi_loss.item() * y.shape[0]
            size = y.shape[-1]
            weights = torch.zeros((size, size), device=device)
            triu_i = torch.triu_indices(size, size, offset=2 + 1)
            weights[triu_i[0], triu_i[1]] = 1
            weights[triu_i[1], triu_i[0]] = 1
            if mask is not None:
                numel = (mask * weights).count_nonzero()
            error = torch.abs((pred - y) * weights).sum() / numel
            error *= y.shape[0]
            total_mae += error
            if idx >= config["sample"] * train_val_ratio:
                break
    total_loss /= idx + 1
    total_mae /= idx + 1
    return total_loss, total_mae


def train_model(
        train_loader,
        val_loader,
        epochs: int,
        config: Dict,
        device: str = None,
        seed: int = 0,
        train_val_ratio: float = 0.8,
        fine_tune: str = None
):
    learning_rate = config["learning_rate"]
    patience = config["patience"]
    torch.manual_seed(seed)
    out_dir = os.path.dirname(config["model_checkpoint"])
    if out_dir != "":
        os.makedirs(out_dir, exist_ok=True)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        assert device.startswith("cuda") or device.startswith("cpu")
    if config["model"] == "normal":
        model = DISTAtteNCionE2(17, nr_updates=config["nr_layers"])
    elif config["model"] == "small":
        model = DISTAtteNCionESmall(17, nr_updates=config["nr_layers"])
    else:
        raise ValueError("no valid Model Type")
    if fine_tune:
        state_dict, old_config = torch.load(fine_tune, map_location="cpu")
        if old_config["model"] != config["model"]:
            raise ValueError("Model type of current configuration does not match the pretrained model type:\n"
                             f"pretrained model: {old_config['model']}\n"
                             f"current model: {config['model']}")
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
    criterion = WeightedDiagonalMSELoss(
        alpha=config["alpha"],
        device=device,
        offset=2,
        # Todo: adjust offset for different min_loop_length
        #  formula: torch.round(mll / 2)
        reduction="sum"
    )
    losses = [(criterion, 1)]
    best_epoch = 0
    best_val_loss = torch.tensor((float("inf")))
    best_val_mae = torch.tensor((float("inf")))
    epoch = 0
    for epoch in range(epochs):
        train_loss = _train(
            model, train_loader, optimizer, device, losses, config
        )
        scheduler.step()
        if not epoch % config["validation_interval"]:
            val_loss, val_mae = _validate(
                model, val_loader, device, losses, config, train_val_ratio
            )
            if val_loss <= best_val_loss:
                best_val_loss = val_loss
                best_val_mae = val_mae
                best_epoch = epoch
                torch.save((model.state_dict(), config), config["model_checkpoint"])
            print(
                f"Epoch: {epoch}\tTraining Loss: {train_loss}\tValidation Loss: {val_loss}\tValidation MAE {val_mae}")
        else:
            print(f"Epoch: {epoch}\tTraining_Loss: {train_loss}")
        if epoch - best_epoch >= patience:
            break
        if torch.isnan(torch.tensor(train_loss)):
            break
    best_val_mae = float(best_val_mae.detach().cpu())
    return {"cost": best_val_mae, "epoch": epoch, "state_dict": model.state_dict()}


def train_network(fasta: str,
                  dataset_path: str,
                  label_dir: str,
                  config: Dict[str, Any],
                  num_threads: int = 1,
                  epochs: int = 400,
                  device: str = None,
                  max_length: int = 200,
                  train_val_ratio: float = 0.8,
                  md_config: Dict = None,
                  mode: str = "normal",
                  seed: int = 0,
                  fine_tune: str = None
                  ):
    """Python API for training a DISTAtteNCionE Network

    Args:
        fasta (str): Path to the Fasta file containing training sequences 
        dataset_path (str): Path where the Dataset object will be stored 
        label_dir (str): Path to the directory created via
            :func:`~RNAdist.NNModels.training_set_generation.rst.training_set_from_fasta`
        config (dict of str): configuration of training process
        num_threads (int): number of parallel processes to use
        epochs (int): maximum number of epochs
        device (str): one of cpu or cuda:x with x specifying the cuda device
        max_length (str): maximum length of the sequences used for padding or window generation
        train_val_ratio (float): part that is used for training. 1-train_val ratio is used for validation
        md_config (dict of str): !!Deprecated!! new versions will infer this from the label_dir
        mode (str): One of "normal" or "window". Specifies the mode that is used for training.
        seed (int): Random number seed for everything related to pytorch
        fine_tune (str): Path to a pretrained model that should be used for fine tuning.

    Examples:
        You can train a network using the following lines  of code:

        >>> from RNAdist.NNModels.training import train_network
        >>> train_network("fasta.fa", "dataset_path", "label_directory")

        You can also change to window mode using a window size of 100 like this

        >>> train_network("fasta.fa", "dataset_path", "label_directory", mode="window", max_length=100)
    """
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
        mode=mode
    )
    train_return = train_model(
        train_loader,
        val_loader,
        epochs,
        config,
        device=device,
        seed=seed,
        train_val_ratio=train_val_ratio,
        fine_tune=fine_tune
    )
    return train_return["state_dict"]



def training_executable_wrapper(args):
    config = {
        "alpha": args.alpha,
        "masking": args.masking,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "validation_interval": args.validation_interval,
        "nr_layers": args.nr_layers,
        "patience": args.patience,
        "optimizer": args.optimizer,
        "model_checkpoint": args.output,
        "lr_step_size": args.learning_rate_step_size,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "model": args.model,
        "gradient_accumulation": args.gradient_accumulation,
        "sample": args.sample

    }

    train_network(
        fasta=args.input,
        label_dir=args.label_dir,
        dataset_path=args.data_path,
        num_threads=args.num_threads,
        max_length=args.max_length,
        config=config,
        seed=args.seed,
        epochs=args.max_epochs,
        device=args.device
    )


if __name__ == '__main__':
    pass
