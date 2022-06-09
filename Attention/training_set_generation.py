import RNA
from sampling.expected_length_sampling import sample_distance
from torch.multiprocessing import Pool
from Bio import SeqIO
import os
import torch
import argparse
import random
from DPModels.viennarna_helpers import set_md_from_config


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]





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
    config_file = os.path.join(output_dir, "config.pt")
    torch.save(index, index_file)
    torch.save(config, config_file)
    return index_file


def mp_wrapper(sequences, file, nr_samples, config):
    md = RNA.md()
    set_md_from_config(md, config)
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


def generation_executable_wrapper(args, md_config):
    training_set_from_fasta(
        args.input,
        args.output,
        md_config,
        num_threads=args.num_threads,
        bin_size=args.bin_size,
        nr_samples=args.nr_samples)


if __name__ == '__main__':
    pass
