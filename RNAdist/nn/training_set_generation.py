import RNA
from RNAdist.sampling.ed_sampling import sample
from torch.multiprocessing import Pool
from Bio import SeqIO
import os
import torch
import random
from RNAdist.dp.viennarna_helpers import set_md_from_config
from typing import Dict, Any

from RNAdist.fasta_wrappers import md_config_from_args


def _chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def training_set_from_fasta(
        fasta: str,
        output_dir: str,
        md_config: Dict[str, Any],
        num_threads: int = 1,
        nr_samples: int = 1000,
        bin_size: int = 100
) -> str:
    """Generates DISTAtteNCionE training set from fasta file

    Args:
        fasta (str): Path to fasta file containing sequences
        output_dir (str): Path to output directory of training set
        md_config (dict of str): configuration dict used to change
            ViennaRNA model details
        num_threads (int): Number of parallel processes that can be used
        nr_samples (int): How many structures per sequence
            should be sampled to generate labels
        bin_size (int): Reduces the number of files written by putting
            labels for bin_size sequences into the same file.

    Returns:
        str: Path to the index file generated in the directory

    Examples:
        >>> training_set_from_fasta("tests/test_data/test.fa", "test_output", num_threads=os.cpu_count())
    """
    to_process = []
    os.makedirs(output_dir, exist_ok=True)
    for seq_record in SeqIO.parse(fasta, "fasta"):
        seq = str(seq_record.seq).upper()
        to_process.append((seq_record.description, seq))
    to_process = list(_chunks(to_process, bin_size))
    files = [os.path.join(output_dir, f"labels_{x}") for x in range(len(to_process))]
    nr_samples = [nr_samples for _ in range(len(to_process))]
    configs = [md_config for _ in range(len(to_process))]
    to_process = list(zip(to_process, files, nr_samples, configs))
    if num_threads <= 1:
        indices = [_mp_wrapper(*call) for call in to_process]
    else:
        with Pool(num_threads) as pool:
            indices = pool.starmap(_mp_wrapper, to_process)
    index = dict(pair for d in indices for pair in d.items())
    index_file = os.path.join(output_dir, "index.pt")
    config_file = os.path.join(output_dir, "config.pt")
    torch.save(index, index_file)
    torch.save(md_config, config_file)
    return index_file


def _mp_wrapper(sequences, file, nr_samples, config):
    md = RNA.md()
    set_md_from_config(md, config)
    out = {}
    index = {}
    for description, seq in sequences:
        y = sample(seq, nr_samples)
        y = torch.tensor(y, dtype=torch.float)
        bppm = torch.tensor(1, dtype=torch.float)
        out[description] = (y, bppm)
        index[description] = os.path.basename(file)
    torch.save(out, file)
    return index


class LabelDict:
    def __init__(self, index_dir):
        self.dir = os.path.abspath(index_dir)
        self.index_file = os.path.join(self.dir, "index.pt")
        self.index = torch.load(self.index_file)
        self.__cache_file = None
        self.__cache = None

    def __getitem__(self, item):
        file = os.path.join(self.dir, self.index[item])
        if file != self.__cache_file:
            self.__cache_file = file
            data = torch.load(file)
            self.__cache = data
        return self.__cache[item]

    def __iter__(self):
        for entry in self.index:
            yield entry

    def items(self):
        for key, value in self.index.items():
            value = self[key]
            yield key, value


def create_random_fasta(outpath: str, nr_sequences: int, seqrange, seed):
    random.seed(seed)
    with open(outpath, "w") as handle:
        for x in range(nr_sequences):
            seq = "".join(random.choices(["A", "C", "U", "G"], k=random.randint(*seqrange)))
            handle.write(f">{x}\n{seq}\n")


def generation_executable_wrapper(args):
    md_config = md_config_from_args(args)
    training_set_from_fasta(
        args.input,
        args.output,
        md_config,
        num_threads=args.num_threads,
        bin_size=args.bin_size,
        nr_samples=args.nr_samples)


if __name__ == '__main__':
    pass


