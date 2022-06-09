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
from RNAdist.Attention.training import (
    train_model,
    dataset_generation,
    loader_generation
)
import numpy as np
from smac.optimizer.multi_objective.parego import ParEGO
from torch.utils.data import random_split


def train_api(config, seed, budget):
    max_epochs = 250
    device = "cuda"
    fasta = "/scratch/ubuntu/RBPdist/Attention/Datasets/generation/unfinished.fasta"
    label_dir = "/scratch/ubuntu/RBPdist/Attention/Datasets/generation/labels/random_80_200"
    dataset_path = "/scratch/pytorch_datasets/rbpdist_cnn_test"
    num_threads = 1
    max_length = 200
    train_val_ratio = 0.2
    total_maes = 0
    total_epochs = 0
    config = dict(config)
    config["validation_interval"] = 5
    config["patience"] = 15 if budget <= 35 else 20
    if "nr_cv" not in config:
        nr_cv = 1 if budget <= 35 else 3
    else:
        nr_cv = config["nr_cv"]

    if "model_checkpoint" not in config:
        config["model_checkpoint"] = "best_models/foo2.pckl"
    config["optimizer"] = torch.optim.Adam if config["optimizer"] == "adam" else torch.optim.SGD
    budget = budget / 100
    torch.manual_seed(seed)
    for _ in range(nr_cv):
        train_set, val_set = dataset_generation(
            fasta=fasta,
            label_dir=label_dir,
            data_storage=dataset_path,
            num_threads=num_threads,
            max_length=max_length,
            train_val_ratio=train_val_ratio
        )
        keep_t = int(len(train_set) * budget)
        keep_val = int(len(val_set) * budget)
        train_set, _ = random_split(
            train_set, [keep_t, len(train_set) - keep_t]
        )
        val_set, _ = random_split(
            val_set, [keep_val, len(val_set) - keep_val]
        )
        train_loader, val_loader = loader_generation(
            train_set, val_set, config["batch_size"]
        )
        costs = train_model(
            train_loader,
            val_loader,
            max_epochs,
            config,
            device=device,
            seed=seed
        )
        total_maes += costs["cost"]
        total_epochs += costs["epoch"]
    total_maes /= nr_cv
    total_epochs /= nr_cv
    return total_maes


if __name__ == '__main__':
    best_model = "best_models/hpo_model_run1.pckl"
    cs = ConfigurationSpace()
    alpha = UniformFloatHyperparameter("alpha", 0.5, 1.0, default_value=0.95)
    masking = CategoricalHyperparameter(
        "masking", [True, False], default_value=True
    )
    learning_rate = UniformFloatHyperparameter(
        "learning_rate", lower=1e-4, upper=1e-2, log=True, default_value=1e-2
    )
    batch_size = CategoricalHyperparameter("batch_size", [8, 16], default_value=16)
    nr_layers = CategoricalHyperparameter("nr_layers", [1, 2])
    optimizer = CategoricalHyperparameter(
        "optimizer",
        ["adam", "sgd"],
        default_value="adam"
    )
    lr_step_size = UniformIntegerHyperparameter(
        "lr_step_size", 10, 100
    )
    cs.add_hyperparameters(
        [alpha,
         masking,
         learning_rate,
         batch_size,
         nr_layers,
         optimizer,
         lr_step_size]
    )

    forbidden_batch_size = CS.ForbiddenEqualsClause(batch_size, 16)
    forbidden_nr_layers = CS.ForbiddenEqualsClause(nr_layers, 2)
    forbidden = CS.ForbiddenAndConjunction(
        forbidden_batch_size, forbidden_nr_layers
    )
    cs.add_forbidden_clause(forbidden)

    scenario = Scenario(
        {
            "run_obj": "quality",
            # we optimize quality (alternative to runtime)
            "cs": cs,  # configuration space
            "deterministic": True,
            "cost_for_crash": [float(MAXINT)],
            "output_dir": "SMAC_OUTPUT",
            "ta_run_limit": 100

        }
    )
    max_budget = 100
    intensifier_kwargs = {"initial_budget": 10, "max_budget": max_budget, "eta": 3}
    smac = SMAC4MF(
        scenario=scenario,
        rng=np.random.RandomState(42),
        tae_runner=train_api,
        multi_objective_algorithm=ParEGO,
        multi_objective_kwargs={
            "rho": 0.05,
        },
        intensifier_kwargs=intensifier_kwargs
    )
    tae = smac.get_tae_runner()

    try:
        incumbent = smac.optimize()
    finally:
        incumbent = smac.solver.incumbent

    incumbent = dict(incumbent)
    incumbent["model"] = best_model
    incumbent["nr_cv"] = 1
    inc_value = tae.run(config=incumbent, budget=max_budget, seed=0)[
        1
    ]

    print("Optimized Value: %.4f" % inc_value)


