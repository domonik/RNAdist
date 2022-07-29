import argparse
from RNAdist.visualize.visualize import run_visualization
from RNAdist.fasta_wrappers import _cp_executable_wrapper, _pmcomp_executable_wrapper, _sampled_distance_executable_wrapper


def add_md_parser(parser):
    group2 = parser.add_argument_group("ViennaRNA Model Details")
    group2.add_argument(
        '--temperature',
        type=float,
        help="Temperature for RNA secondary structure prediction (Default: 37)",
        default=37.0
    )
    group2.add_argument(
        '--min_loop_size',
        type=float,
        help="Minimum Loop size of RNA. (Default: 3)",
        default=3
    )
    group2.add_argument(
        '--noGU',
        type=int,
        help="If set to 1 prevents GU pairs (Default: 0)",
        default=0,
        choices=range(0, 2)
    )
    return parser


def sampling_parser(subparsers, name: str):
    parser = subparsers.add_parser(
        name,
        description='Calculates the averaged distance matrix for every sequence in a '
                    'fasta file using probabilistic backtracking '
    )
    group1 = parser.add_argument_group("General Arguments")
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
        help="Pickled Output file that stores matrices. Can be visualized via RNAdist visualize"
    )
    group1.add_argument(
        '--num_threads',
        type=int,
        help="Number of parallel threads to use (Default: 1)",
        default=1
    )
    group1.add_argument(
        '--nr_samples',
        type=int,
        help="Number of samples used for expected distance calculation. (Default: 1000)",
        default=1000
    )
    parser = add_md_parser(parser)
    return parser


def cp_parser(subparsers, name: str):
    parser = subparsers.add_parser(
        name,
        description=f"Calculates the {name} matrix for every sequence in a fasta file"
    )
    group1 = parser.add_argument_group("General Arguments")
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
        help="Pickled Output file that stores matrices. Can be visualized via RNAdist visualize"
    )
    group1.add_argument(
        '--num_threads',
        type=int,
        help="Number of parallel threads to use (Default: 1)",
        default=1
    )
    parser = add_md_parser(parser)
    return parser


def visualization_parser(subparsers, name):
    parser = subparsers.add_parser(
        name,
        description="Runs the Dash visualization tool"
    )
    parser.add_argument(
        '--input',
        type=str,
        nargs="+",
        help="Path to the prediction files generated using one of the prediction mechansims. It is possible to "
             "include multiple files using whitespace separated paths",
        required=True
    )
    parser.add_argument(
        '--fasta',
        type=str,
        help="Fasta file containing sequences from the prediction. Leavong the default None will lead to missing"
             " nucleotide information (Default: None)",
        default=None
    )
    parser.add_argument(
        '--port',
        type=str,
        help="Port to run the Dash server (Default: 8080)",
        default="8080"
    )
    parser.add_argument(
        '--host',
        type=str,
        help="Host IP used by the dash server to serve the application (Default: 0.0.0.0)",
        default="0.0.0.0"
    )
    return parser


class RNAdistParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            "RNAdist suite",
            usage="RNAdist <command> [<args>]"

        )
        self.methods = {
            "visualize": (visualization_parser, run_visualization),
            "clote-ponty": (cp_parser, _cp_executable_wrapper),
            "pmcomp": (cp_parser, _pmcomp_executable_wrapper),
            "sample": (sampling_parser, _sampled_distance_executable_wrapper)
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


def main():
    RNAdistParser().run()


def rnadist_documentation_wrapper():
    parser = RNAdistParser().parser
    return parser


if __name__ == '__main__':
    main()