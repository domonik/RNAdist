from Attention.Datasets import RNAPairDataset
from Attention.DISTAtteNCionE import (
    DISTAtteNCionE2,
    CovarianceLoss
)
import argparse
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import torch
from typing import Dict, List, Tuple, Callable
import os


def loader_generation(
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


def dataset_generation(
        fasta: str,
        label_dir: str,
        data_storage: str,
        num_threads: int = 1,
        max_length: int = 200,
        train_val_ratio: float = 0.2
):
    dataset = RNAPairDataset(
        data=fasta,
        label_dir=label_dir,
        dataset_path=data_storage,
        num_threads=num_threads,
        max_length=max_length
    )
    t = int(len(dataset) * train_val_ratio)
    v = len(dataset) - t
    training_set, validation_set = random_split(dataset, [t, v])
    return training_set, validation_set


def setup(
        fasta: str,
        label_dir: str,
        data_storage: str,
        batch_size,
        num_threads: int = 1,
        max_length: int = 200,
        train_val_ratio: float = 0.2,
        seed: int = 0
):
    torch.manual_seed(seed)
    train_set, val_set = dataset_generation(
        fasta=fasta,
        label_dir=label_dir,
        data_storage=data_storage,
        num_threads=num_threads,
        max_length=max_length,
        train_val_ratio=train_val_ratio
    )
    train_loader, val_loader = loader_generation(
        train_set,
        val_set,
        batch_size=batch_size,
        num_threads=num_threads
    )

    return train_loader, val_loader


def unpack_batch(batch, device, config):
    if config["masking"]:
        _, pair_rep, y, mask = batch
        mask = mask.to(device)
        numel = torch.count_nonzero(mask)
    else:
        _, pair_rep, y, _ = batch
        numel = y.numel()
        mask = None
    y = y.to(device)
    pair_rep = pair_rep.to(device)
    return pair_rep, y, mask, numel


def train(model, data_loader, optimizer, device,
          losses: List[Tuple[Callable, float]], config: Dict):
    total_loss = 0
    model.train()
    for idx, batch in enumerate(iter(data_loader)):
        optimizer.zero_grad()
        pair_rep, y, mask, numel = unpack_batch(batch, device, config)
        pred = model(pair_rep, mask=mask)
        multi_loss = 0
        for criterion, weight in losses:
            loss = criterion(y, pred) / numel
            multi_loss = multi_loss + loss * weight
        multi_loss.backward()
        optimizer.step()
        total_loss += multi_loss.item() * y.shape[0]
    total_loss /= len(data_loader.dataset)
    return total_loss


def validate(model, data_loader, device, losses: List[Tuple[Callable, float]],
             config):
    total_loss = 0
    total_mae = 0
    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(iter(data_loader)):
            pair_rep, y, mask, numel = unpack_batch(batch, device, config)
            pred = model(pair_rep, mask=mask)
            multi_loss = 0
            for criterion, weight in losses:
                loss = criterion(y, pred) / numel
                multi_loss = multi_loss + loss * weight
            total_loss += multi_loss.item() * y.shape[0]
            error = torch.abs(pred - y).sum() / y.numel()
            error *= y.shape[0]
            total_mae += error
    total_loss /= len(data_loader.dataset)
    total_mae /= len(data_loader.dataset)
    return total_loss, total_mae


def train_model(
        train_loader,
        val_loader,
        epochs: int,
        config: Dict,
        device: str = None,
        seed: int = 0,
):
    learning_rate = config["learning_rate"]
    patience = config["patience"]
    torch.manual_seed(seed)
    os.makedirs(os.path.dirname(config["model_checkpoint"]), exist_ok=True)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        assert device.startswith("cuda") or device.startswith("cpu")
    model = DISTAtteNCionE2(17, nr_updates=config["nr_layers"])
    model.to(device)
    optimizer = config["optimizer"](model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=config["lr_step_size"], gamma=0.1
    )
    criterion = torch.nn.MSELoss(reduction="sum")
    criterion2 = CovarianceLoss(reduction="sum")
    losses = [(criterion, config["alpha"]), (criterion2, 1 - config["alpha"])]
    best_epoch = 0
    best_val_mae = torch.tensor((float("inf")))
    epoch = 0
    for epoch in range(epochs):
        train_loss = train(
            model, train_loader, optimizer, device, losses, config
        )
        scheduler.step()
        if not epoch % config["validation_interval"]:
            val_loss, val_mae = validate(
                model, val_loader, device, losses, config
            )
            if val_mae <= best_val_mae:
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
    return {"cost": best_val_mae, "epoch": epoch}


def main(fasta, data_path, label_dir, config, num_threads: int = 1,
         epochs: int = 400, device=None, max_length: int = 200,
         train_val_ratio: float = 0.2, seed: int = 0):
    train_loader, val_loader = setup(
        fasta=fasta,
        label_dir=label_dir,
        data_storage=data_path,
        batch_size=config["batch_size"],
        num_threads=num_threads,
        max_length=max_length,
        train_val_ratio=train_val_ratio,
        seed=seed
    )
    best_val_mae = train_model(
        train_loader,
        val_loader,
        epochs,
        config,
        device=device,
        seed=seed
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Trains DISTAttenCionE NeuralNetwork'
    )
    group1 = parser.add_argument_group("Dataset Arguments")
    group2 = parser.add_argument_group("Model Configuration")
    group3 = parser.add_argument_group(
        "ViennaRNA Model Details",
        description="Model details are inferred from the training set "
                    "and overwritten in new versions"
    )
    group1.add_argument(
        '--input',
        metavar='i',
        type=str,
        help="FASTA input file",
        required=True
    )
    group1.add_argument(
        '--label_dir',
        metavar='l',
        required=True,
        type=str,
        help="path to directory that was generated via "
             "training_set_generation.py"
    )
    group1.add_argument(
        '--output',
        metavar='o',
        required=True,
        type=str,
        help="path to the model file that will be generated"
    )
    group1.add_argument(
        '--data_path',
        metavar='dp',
        type=str,
        required=True,
        help="Directory to store the pickled Dataset. "
             "It is created if it does not exist yet",
    )
    group1.add_argument(
        '--num_threads',
        metavar='t',
        type=int,
        help="Number of parallel threads to use (Default: 1)",
        default=1
    )
    group1.add_argument(
        '--seed',
        metavar='s',
        type=int,
        help="Random Number Seed to be used (Default: 0)",
        default=0
    )
    group2.add_argument(
        '--max_length',
        metavar='ml',
        type=int,
        help="Maximum length of RNAs. (Default: 200)",
        default=200
    )
    group2.add_argument(
        '--alpha',
        metavar='a',
        type=float,
        help="weight for the MSE part of the combined loss."
             "The Covariance Part will be weighted 1 - alpha (Default: 1)",
        default=1
    )
    group2.add_argument(
        '--masking',
        metavar='m',
        type=bool,
        help="whether masking is applied during training (Default: True)",
        default=True
    )
    group2.add_argument(
        '--learning_rate',
        metavar='lr',
        type=float,
        help="Initial Learning Rate (Default: 0.001)",
        default=0.001
    )
    group2.add_argument(
        '--batch_size',
        metavar='bs',
        type=int,
        help="Batch Size (Default: 16)",
        default=16
    )
    group2.add_argument(
        '--max_epochs',
        metavar='me',
        type=int,
        help="Maximum number of epochs to train (Default: 400)",
        default=400
    )
    group2.add_argument(
        '--validation_interval',
        metavar='vi',
        type=int,
        help="Specifies after how many epochs validation should be done",
        default=5
    )
    group2.add_argument(
        '--nr_layers',
        metavar='nl',
        type=int,
        help="Number of Pair Representation Update Layers that should be "
             "stacked. (Default: 1)",
        default=1
    )
    group2.add_argument(
        '--patience',
        metavar='pt',
        type=int,
        help="Patience of the training procedure (Default: 20)",
        default=20
    )
    group2.add_argument(
        '--optimizer',
        metavar='opt',
        type=str,
        help="Optimizer that should be used. Can be either adam or sgd."
             "(Default: adam)",
        default="adam"
    )
    group2.add_argument(
        '--learning_rate_step_size',
        metavar='lrss',
        type=int,
        help="Decreases learning rate by 0.1 * current lr after the specified"
             "nr of epochs (Default: 50)",
        default=50
    )
    group3.add_argument(
        '--temperature',
        metavar='s',
        type=float,
        help="Temperature for RNA secondary structure prediction (Default: 37)",
        default=37.0
    )
    group3.add_argument(
        '--min_loop_size',
        metavar='s',
        type=float,
        help="Minimum Loop size of RNA. (Default: 3)",
        default=3
    )
    group3.add_argument(
        '--noGU',
        metavar='s',
        type=int,
        help="If set to 1 prevents GU pairs (Default: 0)",
        default=0,
        choices=range(0, 2)
    )
    args = parser.parse_args()
    md_config = {
        "temperature": args.temperature,
        "min_loop_size": args.min_loop_size,
        "noGU": args.noGU,
    }
    if args.optimizer.lower() == "sgd":
        opt = torch.optim.SGD
    else:
        opt = torch.optim.Adam
    config = {
        "alpha": args.alpha,
        "masking": args.masking,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "validation_interval": args.validation_interval,
        "nr_layers": args.nr_layers,
        "patience": args.patience,
        "optimizer": opt,
        "model_checkpoint": args.output,
        "lr_step_size": args.learning_rate_step_size

    }

    main(
        fasta=args.input,
        label_dir=args.label_dir,
        data_path=args.data_path,
        num_threads=args.num_threads,
        max_length=args.max_length,
        config=config,
        seed=args.seed,
        epochs=args.max_epochs
    )
