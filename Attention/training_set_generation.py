import RNA
from sampling.expected_length_sampling import sample_distance
from torch.multiprocessing import Pool
from Bio import SeqIO
import os
import torch
import argparse
import random


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def set_md(md, config):
    for key, value in config.items():
        setattr(md, key, value)


def training_set_from_fasta(fasta, output_dir, config, num_threads: int = 1, nr_samples: int = 1000, bin_size: int = 100):
    to_process = []
    os.makedirs(output_dir, exist_ok=True)
    for seq_record in SeqIO.parse(fasta, "fasta"):
        seq = str(seq_record.seq).upper()
        to_process.append((seq_record.description, seq))
    to_process = list(chunks(to_process, bin_size))
    files = [os.path.join(output_dir, f"labels_{x}") for x in range(len(to_process))]
    nr_samples = [nr_samples for _ in range(len(to_process))]
    configs = [config for _ in range(len(to_process))]
    to_process = list(zip(to_process, files, nr_samples, configs))
    if num_threads <= 1:
        indices = [mp_wrapper(*call) for call in to_process]
    else:
        with Pool(num_threads) as pool:
            indices = pool.starmap(mp_wrapper, to_process)
    index = dict(pair for d in indices for pair in d.items())
    index_file = os.path.join(output_dir, "index.pt")
    torch.save(index, index_file)
    return index_file


def mp_wrapper(sequences, file, nr_samples, config):
    md = RNA.md()
    set_md(md, config)
    out = {}
    index = {}
    for description, seq in sequences:
        y, bppm = sample_distance(seq, nr_samples)
        y = torch.tensor(y, dtype=torch.float)
        bppm = torch.tensor(bppm, dtype=torch.float)
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




if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate DISTAttenCionE training set'
    )
    group1 = parser.add_argument_group("Dataset Generation")
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
    args = parser.parse_args()
    config = {
        "temperature": args.temperature,
        "min_loop_size": args.min_loop_size,
        "noGU": args.noGU,
    }

    p = 0
    training_set_from_fasta(args.input, args.output, config, num_threads=args.num_threads, bin_size=args.bin_size, nr_samples=args.nr_samples)