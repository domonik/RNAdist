import ConfigSpace as CS
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)
from smac.configspace import ConfigurationSpace
import torch
from smac.facade.smac_mf_facade import SMAC4MF
from smac.scenario.scenario import Scenario
from smac.utils.constants import MAXINT
from RNAdist.nn.training import (
    train_model,
    _dataset_generation,
    _loader_generation
)
from RNAdist.nn.configuration import ModelConfiguration
import numpy as np
from torch.utils.data import random_split
from tempfile import TemporaryDirectory
from ConfigSpace.conditions import InCondition
import os


class TrainingWorker:
    def __init__(self, fasta, label_dir, dataset_path, num_threads, max_length,
                 train_val_ratio, max_epochs, device):
        self.fasta = fasta
        self.label_dir = label_dir
        self.dataset_path = dataset_path
        self.num_threads = num_threads
        self.max_length = max_length
        self.device = device
        self.train_val_ratio = train_val_ratio
        self.max_epochs = max_epochs

    def partial_set(self, train_set, val_set, budget):
        keep_t = max(int(len(train_set) * budget), 1)
        keep_val = max(int(len(val_set) * budget), 1)
        train_set, _ = random_split(
            train_set, [keep_t, len(train_set) - keep_t]
        )
        val_set, _ = random_split(
            val_set, [keep_val, len(val_set) - keep_val]
        )
        return train_set, val_set

    def train_api(self, config, seed, budget):
        config = dict(config)
        config["patience"] = 10 if budget <= 35 else 20
        if "model_checkpoint" not in config:
            tmpdir = TemporaryDirectory(prefix="SMAC_RNAdist_")
            checkpoint = os.path.join(tmpdir.name, "model.pckl")
            config["model_checkpoint"] = checkpoint
        else:
            tmpdir = None
        if "lr_step_size" not in config:
            config["lr_step_size"] = None
        if "momentum" not in config:
            config["momentum"] = None

        mdc = ModelConfiguration(
            model_checkpoint=config["model_checkpoint"],
            model=config["model_type"],
            optimizer=config["optimizer"],
            learning_rate=config["learning_rate"],
            batch_size=config["batch_size"],
            nr_layers=config["nr_layers"],
            validation_interval=1,
            patience=config["patience"],
            lr_step_size=config["lr_step_size"],
            momentum=config["momentum"],
            weight_decay=config["weight_decay"],
            random_shift=config["random_shift"],
            normalize_bpp=config["normalize_bpp"]
        )
        budget = budget / 100
        torch.manual_seed(seed)
        train_set, val_set = _dataset_generation(
            fasta=self.fasta,
            label_dir=self.label_dir,
            data_storage=self.dataset_path,
            num_threads=self.num_threads,
            max_length=self.max_length,
            train_val_ratio=self.train_val_ratio
        )
        train_set, val_set = self.partial_set(train_set, val_set, budget)
        train_loader, val_loader = _loader_generation(
            train_set, val_set, config["batch_size"]
        )
        costs = train_model(
            train_loader,
            val_loader,
            self.max_epochs,
            mdc,
            device=self.device,
            seed=seed
        )
        if tmpdir:
            tmpdir.cleanup()
        return costs["cost"]


def smac_that(
        fasta,
        model_output,
        smac_dir,
        label_dir,
        dataset_path,
        max_length,
        train_val_ratio: float = 0.2,
        device: str = "cuda",
        max_epochs: int = 200,
        num_threads: int = 1,
        run_default: bool = False,
        ta_run_limit: int = 100
):
    cs = ConfigurationSpace()
    learning_rate = UniformFloatHyperparameter(
        "learning_rate", lower=1e-4, upper=1e-2, log=True, default_value=1e-2
    )
    batch_size = CategoricalHyperparameter("batch_size", [16, 8])
    nr_layers = CategoricalHyperparameter("nr_layers", [1, 2])
    optimizer = CategoricalHyperparameter(
            "optimizer",
            ["adamw", "sgd"],
        )
    model_type = CategoricalHyperparameter(
            "model_type",
            ["normal", "small"],
        )
    random_shift = CategoricalHyperparameter(
        "random_shift",
        [0, 0.5, 1],
    )
    normalize_bpp = CategoricalHyperparameter(
        "normalize_bpp",
        [True, False],
    )
    lr_step_size = UniformIntegerHyperparameter(
        "lr_step_size", 10, 100
    )
    momentum = UniformFloatHyperparameter(
        "momentum", 0.5, 0.99
    )
    weight_decay = UniformFloatHyperparameter(
        "weight_decay", 0.0001, 0.1,  log=True
    )
    cs.add_hyperparameters(
        [
            random_shift,
            normalize_bpp,
            model_type,
            learning_rate,
            batch_size,
            nr_layers,
            optimizer,
            lr_step_size,
            momentum,
            weight_decay
        ]
    )
    forbidden_batch_size = CS.ForbiddenEqualsClause(batch_size, 16)
    forbidden_nr_layers = CS.ForbiddenEqualsClause(nr_layers, 2)
    forbidden = CS.ForbiddenAndConjunction(
        forbidden_batch_size, forbidden_nr_layers
    )
    cs.add_forbidden_clause(forbidden)
    cs.add_condition(
        InCondition(child=momentum, parent=optimizer, values=["sgd"])
    )
    cs.add_condition(
        InCondition(child=lr_step_size, parent=optimizer, values=["sgd"])
    )
    scenario = Scenario(
        {
            "run_obj": "quality",
            # we optimize quality (alternative to runtime)
            "cs": cs,  # configuration space
            "deterministic": True,
            "cost_for_crash": [float(MAXINT)],
            "output_dir": smac_dir,
            "ta_run_limit": ta_run_limit

        }
    )
    max_budget = 100
    intensifier_kwargs = {
        "initial_budget": 10,
        "max_budget": max_budget,
        "eta": 3
    }
    worker = TrainingWorker(
        fasta=fasta,
        label_dir=label_dir,
        dataset_path=dataset_path,
        num_threads=num_threads,
        max_length=max_length,
        train_val_ratio=train_val_ratio,
        device=device,
        max_epochs=max_epochs
    )
    smac = SMAC4MF(
        scenario=scenario,
        rng=np.random.RandomState(42),
        tae_runner=worker.train_api,
        intensifier_kwargs=intensifier_kwargs
    )
    tae = smac.get_tae_runner()
    if run_default:
        def_value = tae.run(
            config=cs.get_default_configuration(),
            budget=10, seed=0
        )[1]
    try:
        incumbent = smac.optimize()
    finally:
        incumbent = smac.solver.incumbent

    incumbent = dict(incumbent)
    incumbent["model_checkpoint"] = model_output
    inc_run = tae.run(config=incumbent, budget=max_budget, seed=0)
    inc_value = inc_run[1]
    assert inc_run[0].name == "SUCCESS", "Incumbent Run Failed with error"
    print("Optimized Value: %.4f" % inc_value)
    print(f"Saved optimized model in: {model_output}")


def smac_executable_wrapper(args):
    smac_that(
        fasta=args.input,
        model_output=args.output,
        label_dir=args.label_dir,
        dataset_path=args.dataset_path,
        max_length=args.max_length,
        train_val_ratio=args.train_val_ratio,
        device=args.device,
        max_epochs=args.max_epochs,
        num_threads=args.num_threads,
        run_default=args.run_default,
        smac_dir=args.smac_dir,
        ta_run_limit=args.ta_run_limit
    )


if __name__ == '__main__':
    pass
