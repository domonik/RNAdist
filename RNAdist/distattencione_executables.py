import argparse
import math

from RNAdist.nn.prediction import prediction_executable_wrapper
from RNAdist.nn.smac_optimize import smac_executable_wrapper
from RNAdist.nn.training import training_executable_wrapper
from RNAdist.nn.training_set_generation import generation_executable_wrapper
from RNAdist.executables import add_md_parser


class DISTAtteNCioNEParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            "DISTAtteNCioNE suite",
            usage="DISTAtteNCioNE <command> [<args>]"

        )
        self.methods = {
            "generate_data": (generation_parser, generation_executable_wrapper),
            "train": (training_parser, training_executable_wrapper),
            "predict": (prediction_parser, prediction_executable_wrapper),
            "optimize": (smac_parser, smac_executable_wrapper)
        }
        self.subparsers = self.parser.add_subparsers()
        self.__addparsers()

    def __addparsers(self):
        for name, (parser_add, func) in self.methods.items():
            subp = parser_add(self.subparsers, name)
            subp.set_defaults(func=func)

    def parse_args(self):
        args = self.parser.parse_args()
        return args

    def run(self):
        args = self.parse_args()
        args.func(args)


def smac_parser(subparsers, name: str):
    parser = subparsers.add_parser(
        name,
        description="Runs SMAC optimization for a DISTAtteNCionE Network"
    )
    group1 = parser.add_argument_group("Trainig Data")
    group2 = parser.add_argument_group("Training Settings")
    group1.add_argument(
        '--input',
        type=str,
        help="FASTA File used for training set generation",
        required=True
    )
    group1.add_argument(
        '--output',
        type=str,
        help="Path where the optimized model will be saved",
        required=True
    )
    group1.add_argument(
        '--label_dir',
        type=str,
        help="Output directory of training data generation",
        required=True
    )
    group1.add_argument(
        '--dataset_path',
        type=str,
        help="Path where the Pytorch Dataset will be stored",
        required=True
    )
    group2.add_argument(
        '--smac_dir',
        type=str,
        help="Path where the smac optimization data is stored"
             " (Default: SMAC_OUTPUT)",
        default="SMAC_OUTPUT"
    )
    group2.add_argument(
        '--max_length',
        type=int,
        help="Maximum Length of the RNA in the FASTA File (Default: 200)",
        default=200
    )
    group2.add_argument(
        '--train_val_ratio',
        type=float,
        help="Split ratio for Training, Validation split (Default: 0.2)",
        default=0.2
    )
    group2.add_argument(
        '--device',
        type=str,
        help="Device to train on (Default: cuda) "
             "It is not recommended to use CPU for HPO",
        default="cuda"
    )
    group2.add_argument(
        '--max_epochs',
        type=int,
        help="Maximum nr of epochs to train (Default: 200)",
        default=200
    )
    group2.add_argument(
        '--num_threads',
        type=int,
        help="Maximum nr of cores used for training (Default: 1)",
        default=1
    )
    group2.add_argument(
        '--run_default',
        help="Whether to run default HPO settings before optimization"
             " (Default: False)",
        default=False,
        action="store_true"
    )
    group2.add_argument(
        '--ta_run_limit',
        type=int,
        help="Maximum nr of hyper parameter configurations tested (Default: 1)",
        default=100
    )
    return parser


def training_parser(subparsers, name):
    parser = subparsers.add_parser(
        name,
        description='Trains DISTAttenCionE NeuralNetwork'
    )
    group1 = parser.add_argument_group("Dataset Arguments")
    group2 = parser.add_argument_group("Model Configuration")
    group1.add_argument(
        '--input',
        type=str,
        help="FASTA input file",
        required=True
    )
    group1.add_argument(
        '--label_dir',
        required=True,
        type=str,
        help="path to directory that was generated via "
             "training_set_generation.py"
    )
    group1.add_argument(
        '--output',
        required=True,
        type=str,
        help="path to the model file that will be generated"
    )
    group1.add_argument(
        '--dataset_path',
        type=str,
        required=True,
        help="Directory to store the pickled Dataset. "
             "It is created if it does not exist yet",
    )
    group1.add_argument(
        '--num_threads',
        type=int,
        help="Number of parallel threads to use (Default: 1)",
        default=1
    )
    group1.add_argument(
        '--seed',
        type=int,
        help="Random Number Seed to be used (Default: 0)",
        default=0
    )
    group2.add_argument(
        '--model',
        type=str,
        help="Model to train: choose between normal, small",
        default="normal"
    )
    group2.add_argument(
        '--fine_tune',
        type=str,
        help="Path to pretrained model that should be used in fine_tuning (Default: None)",
        default=None
    )
    group2.add_argument(
        '--max_length',
        type=int,
        help="Maximum length of RNAs. (Default: 200)",
        default=200
    )
    group2.add_argument(
        '--no_masking',
        help="No masking is applied during training",
        default=False,
        action="store_true"

    )
    group2.add_argument(
        '--sample',
        type=int,
        help="Number of samples drawn from training and validation set (with replacement)",
        default=None
    )
    group2.add_argument(
        '--learning_rate',
        type=float,
        help="Initial Learning Rate (Default: 0.001)",
        default=0.001
    )
    group2.add_argument(
        '--batch_size',
        type=int,
        help="Batch Size (Default: 16)",
        default=16
    )
    group2.add_argument(
        '--gradient_accumulation',
        type=int,
        help="Update weights after n mini batches. (Default: 1)",
        default=1
    )
    group2.add_argument(
        '--max_epochs',
        type=int,
        help="Maximum number of epochs to train (Default: 400)",
        default=400
    )
    group2.add_argument(
        '--validation_interval',
        type=int,
        help="Specifies after how many epochs validation should be done",
        default=5
    )
    group2.add_argument(
        '--nr_layers',
        type=int,
        help="Number of Pair Representation Update Layers that should be "
             "stacked. (Default: 1)",
        default=1
    )
    group2.add_argument(
        '--patience',
        type=int,
        help="Patience of the training procedure (Default: 20)",
        default=20
    )
    group2.add_argument(
        '--optimizer',
        type=str,
        help="Optimizer that should be used. Can be either AdamW or SGD."
             "(Default: AdamW)",
        default="AdamW"
    )
    group2.add_argument(
        '--momentum',
        type=float,
        help="Momentum for sgd is ignored if optimizer is adamw (Default: 0).",
        default=0
    )
    group2.add_argument(
        '--weight_decay',
        type=float,
        help="Weight decay (Default: 0)",
        default=0
    )
    group2.add_argument(
        '--learning_rate_step_size',
        type=int,
        help="Decreases learning rate by 0.1 * current lr after the specified"
             "nr of epochs. Only used if optimizer is SGD (Default: 50)",
        default=50
    )
    group2.add_argument(
        '--device',
        type=str,
        help="device to run prediction on (Default: "
             "automatically determines if gpu is available)",
        default=None
    )
    group2.add_argument(
        '--exclude_bppm',
        help="Whether to use the basepair probability matrix as a feature",
        default=False,
        action="store_true"
    )
    group2.add_argument(
        '--exclude_position',
        help="Excludes the position encoding as a feature",
        default=False,
        action="store_true"
    )
    group2.add_argument(
        '--random_shift',
        help="Applies a random shift in indices of nucleotides",
        default=None,
        type=float
    )
    group2.add_argument(
        '--normalize_bppm',
        help="Min Max normalizes the base pairing probability matrix",
        default=False,
        action="store_true"
    )
    group2.add_argument(
        '--gradient_checkpointing',
        help="Enables gradient checkpointing to reduce memory consumption",
        default=False,
        action="store_true"
    )
    return parser


def generation_parser(subparsers, name: str):
    parser = subparsers.add_parser(
        name,
        description='Generate DISTAttenCionE training set'
    )
    group1 = parser.add_argument_group("Dataset Generation")
    group1.add_argument(
        '--input',
        type=str,
        help="FASTA input file",
        required=True
    )
    group1.add_argument(
        '--output',
        required=True,
        type=str,
        help="Output Directory. It is created automatically "
             "if it does not exist yet"
    )
    group1.add_argument(
        '--num_threads',
        type=int,
        help="Number of parallel threads to use (Default: 1)",
        default=1
    )
    group1.add_argument(
        '--bin_size',
        type=int,
        help="Number of sequences stored in a single file. (Default: 1000)",
        default=1000
    )
    group1.add_argument(
        '--nr_samples',
        type=int,
        help="Number of samples used for expected distance calculation. (Default: 1000)",
        default=1000
    )
    parser = add_md_parser(parser)
    return parser


def prediction_parser(subparsers, name: str):
    parser = subparsers.add_parser(
        name,
        description="Predicts Expected Distances using DISTAtteNCionE model"
    )
    group1 = parser.add_argument_group("Prediction")
    group1.add_argument(
        '--input',
        type=str,
        help="FASTA input file",
        required=True
    )
    group1.add_argument(
        '--output',
        required=True,
        type=str,
        help="Output File will be pickled file "
             "containing Expected Distances as numpy arrays"
    )
    group1.add_argument(
        '--model_file',
        required=True,
        type=str,
        help="Path to the trained model"
    )
    group1.add_argument(
        '--num_threads',
        type=int,
        help="Number of parallel threads to use (Default: 1)",
        default=1
    )
    group1.add_argument(
        '--batch_size',
        type=int,
        default=1,
        help="Batch Size for prediction"
    )
    group1.add_argument(
        '--device',
        type=str,
        help="device to run prediction on (Default: cpu)",
        default="cpu"
    )
    group1.add_argument(
        '--max_length',
        type=int,
        default=1,
        help="Maximum length for padding of the RNAs"
    )
    parser = add_md_parser(parser)
    return parser


def documentation_wrapper():
    parser = DISTAtteNCioNEParser().parser
    return parser


def main():
    DISTAtteNCioNEParser().run()


if __name__ == '__main__':
    main()