import argparse
import sys

from RNAdist.Attention.prediction import prediction_executable_wrapper
from RNAdist.Attention.training import training_executable_wrapper
from RNAdist.Attention.training_set_generation import (
    generation_executable_wrapper
)


def training_parser():
    parser = argparse.ArgumentParser(
        description='Trains DISTAttenCionE NeuralNetwork'
    )
    group1 = parser.add_argument_group("Dataset Arguments")
    group2 = parser.add_argument_group("Model Configuration")
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
    return parser


def generation_parser():
    parser = argparse.ArgumentParser(
        description='Generate DISTAttenCionE training set'
    )
    group1 = parser.add_argument_group("Dataset Generation")
    group1.add_argument(
        '--input',
        metavar='i',
        type=str,
        help="FASTA input file",
        required=True
    )
    group1.add_argument(
        '--output',
        metavar='o',
        required=True,
        type=str,
        help="Output Directory. It is created automatically "
             "if it does not exist yet"
    )
    group1.add_argument(
        '--num_threads',
        metavar='t',
        type=int,
        help="Number of parallel threads to use (Default: 1)",
        default=1
    )
    group1.add_argument(
        '--bin_size',
        metavar='b',
        type=int,
        help="Number of sequences stored in a single file. (Default: 1000)",
        default=1000
    )
    group1.add_argument(
        '--nr_samples',
        metavar='s',
        type=int,
        help="Number of samples used for expected distance calculation. (Default: 1000)",
        default=1000
    )

    return parser


def prediction_parser():
    parser = argparse.ArgumentParser(
        description="Predicts Expected Distances using DISTAtteNCionE model"
    )
    group1 = parser.add_argument_group("Prediction")
    group2 = parser.add_argument_group("Model Details")
    group1.add_argument(
        '--input',
        metavar='i',
        type=str,
        help="FASTA input file",
        required=True
    )
    group1.add_argument(
        '--output',
        metavar='o',
        required=True,
        type=str,
        help="Output File will be pickled file "
             "containing Expected Distances as numpy arrays"
    )
    group1.add_argument(
        '--model_file',
        metavar='o',
        required=True,
        type=str,
        help="Path to the trained model"
    )
    group1.add_argument(
        '--num_threads',
        metavar='t',
        type=int,
        help="Number of parallel threads to use (Default: 1)",
        default=1
    )
    group1.add_argument(
        '--batch_size',
        metavar='bs',
        type=int,
        default=1,
        help="Batch Size for prediction"
    )
    group1.add_argument(
        '--device',
        metavar='d',
        type=str,
        help="device to run prediction on (Default: cpu)",
        default="cpu"
    )
    group1.add_argument(
        '--max_length',
        metavar='ml',
        type=int,
        default=1,
        help="Maximum length for padding of the RNAs"
    )
    return parser

def add_md_parser(parser):
    group2 = parser.add_argument_group("ViennaRNA Model Details")
    group2.add_argument(
        '--temperature',
        metavar='s',
        type=float,
        help="Temperature for RNA secondary structure prediction (Default: 37)",
        default=37.0
    )
    group2.add_argument(
        '--min_loop_size',
        metavar='s',
        type=float,
        help="Minimum Loop size of RNA. (Default: 3)",
        default=3
    )
    group2.add_argument(
        '--noGU',
        metavar='s',
        type=int,
        help="If set to 1 prevents GU pairs (Default: 0)",
        default=0,
        choices=range(0, 2)
    )
    return parser

def md_config_from_args(args):
    md_config = {
        "temperature": args.temperature,
        "min_loop_size": args.min_loop_size,
        "noGU": args.noGU,
    }
    return md_config


class DISTAtteNCioNEParser:
    def __init__(self):
        parser = argparse.ArgumentParser(
            "DISTAtteNCioNE suite",
            usage="DISTAtteNCioNE <command> [<args>]"

        )
        self.__object_methods = self.__get_modes()

        help_methods = ", ".join(self.__object_methods)
        help_msg = f"one of: {help_methods}"
        parser.add_argument("command", help=help_msg)
        args = parser.parse_args(sys.argv[1:2])
        if args.command not in self.__object_methods:
            print('Unrecognized command')
            parser.print_help()
            exit(1)
        getattr(self, args.command)()

    def __get_modes(self):
        object_methods = []
        for method_name in dir(DISTAtteNCioNEParser):
            if callable(getattr(DISTAtteNCioNEParser, method_name)):
                if not method_name.startswith("_"):
                    object_methods.append(method_name)
        return object_methods

    def train(self):
        parser = training_parser()
        parser = add_md_parser(parser)
        args = parser.parse_args(sys.argv[2:])
        md_config = md_config_from_args(args)
        training_executable_wrapper(args, md_config)

    def generate_data(self):
        parser = generation_parser()
        parser = add_md_parser(parser)
        args = parser.parse_args(sys.argv[2:])
        md_config = md_config_from_args(args)
        generation_executable_wrapper(args, md_config)

    def predict(self):
        parser = prediction_parser()
        parser = add_md_parser(parser)
        args = parser.parse_args(sys.argv[2:])
        md_config = md_config_from_args(args)
        prediction_executable_wrapper(args, md_config)



def main():
    DISTAtteNCioNEParser()



if __name__ == '__main__':
    main()