import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import os
from RNAdist.Attention.training_set_generation import LabelDict
from functools import cached_property
from typing import List, Union
from Bio import SeqIO
from torch.multiprocessing import Pool
import RNA
import math
import torch.multiprocessing
from RNAdist.DPModels.viennarna_helpers import fold_bppm, set_md_from_config


NUCLEOTIDE_MAPPING = {
    "A": [1, 0, 0, 0],
    "U": [0, 1, 0, 0],
    "C": [0, 0, 1, 0],
    "G": [0, 0, 0, 1]
}


class RNADATA():
    def __init__(self, sequence, description=None, md=None):
        self.sequence = sequence.replace("T", "U")
        self.description = description
        self.md = md

    @staticmethod
    def pos_encoding(idx, dimension):
        enc = []
        for dim in range(int(dimension / 2)):
            w = 1 / (10000 ** (2 * dim / dimension))
            sin = math.sin(w * idx)
            cos = math.cos(w * idx)
            enc += [sin, cos]
        return enc

    def to_tensor(self, max_length, positional_encoding, pos_encoding_dim=4):
        bppm = fold_bppm(self.sequence, self.md)
        bppm = torch.tensor(bppm, dtype=torch.float)
        bppm = self._add_upprob(bppm)
        n = bppm.shape[0]
        seq_embedding = [NUCLEOTIDE_MAPPING[m][:] for m in self.sequence]
        if positional_encoding:
            for idx, letter in enumerate(seq_embedding):
                enc = self.pos_encoding(idx, pos_encoding_dim)
                seq_embedding[idx] += enc
        pair_matrix = bppm[:, :, None]
        seq_embedding = torch.tensor(seq_embedding, dtype=torch.float)
        if seq_embedding.shape[0] < max_length:
            pad_val = max_length - seq_embedding.shape[0]
            seq_embedding = F.pad(seq_embedding, (0, 0, 0, pad_val),
                                  "constant", 0)
            pair_matrix = F.pad(pair_matrix, (0, 0, 0, pad_val, 0, pad_val),
                                "constant", 0)
        return pair_matrix, seq_embedding

    @staticmethod
    def _add_upprob(bppm):
        for idx1 in range(bppm.shape[0]):
            upprob = 1 - bppm[idx1].sum().item()
            if idx1 != 0 and idx1 != bppm.shape[0] - 1:
                upprob = upprob / 2
            for idx2 in range(bppm.shape[0]):
                if idx2 == idx1 + 1 or idx2 == idx1 - 1:
                    bppm[idx1, idx2] = upprob
        return bppm


class RNAPairDataset(Dataset):
    def __init__(self, data: str,
                 label_dir: Union[str, os.PathLike, None],
                 dataset_path: str = "./",
                 num_threads: int = 1,
                 max_length: int = 200,
                 md_config=None):
        super().__init__()
        if not os.path.exists(dataset_path):
            os.makedirs(dataset_path, exist_ok=True)
        self.dataset_path = dataset_path
        self.data = data
        self.extension = "_data.pt"
        self.num_threads = num_threads
        self.max_length = max_length
        self.label_file = label_dir
        self.md_config = md_config if md_config is not None else {}
        if label_dir is not None:
            self.label_dict = LabelDict(label_dir)
            md_config_file = os.path.join(label_dir, "config.pt")
            if os.path.exists(md_config_file):
                md_config = torch.load(md_config_file)
                self.md_config = md_config
                print("setting model details from training configuration")
            else:
                print("Not able to infer model details from set generation "
                      "output. Make sure to set them correctly by hand")
        else:
            self.label_dict = None
        if not self._dataset_generated(self._files):
            self.generate_dataset()

    @staticmethod
    def _check_input_files_exist(files: List[str]):
        for file in files:
            assert os.path.exists(file)

    @staticmethod
    def _dataset_generated(files: List[str]):
        for element in files:
            if not os.path.exists(element):
                return False
        return True

    @cached_property
    def rna_graphs(self):
        data = []
        descriptions = set()
        for seq_record in SeqIO.parse(self.data, "fasta"):
            assert seq_record.description not in descriptions, "Fasta headers must be unique"
            data.append((seq_record.description, str(seq_record.seq)))
            descriptions.add(seq_record.description)
        return data

    @cached_property
    def _files(self):
        files = []
        for idx, seq_data in enumerate(self.rna_graphs):
            files.append(
                os.path.join(self.dataset_path, f"{idx}{self.extension}"))
        return files

    def generate_dataset(self):
        l = [self.label_dict for _ in range(len(self._files))]
        ml = [self.max_length for _ in range(len(self._files))]
        mds = [self.md_config for _ in range(len(self._files))]
        calls = list(zip(self._files, self.rna_graphs, ml, l, mds))
        if self.num_threads == 1:
            for call in calls:
                self.mp_create_wrapper(*call)
        else:
            with Pool(self.num_threads) as pool:
                pool.starmap(self.mp_create_wrapper, calls)

    @staticmethod
    def mp_create_wrapper(file, seq_data, max_length, label_dict, md_config):
        description, seq = seq_data
        md = RNA.md()
        set_md_from_config(md, md_config)
        rna_data = RNADATA(seq, description, md)
        pair_matrix, seq_embedding = rna_data.to_tensor(
            max_length,
            positional_encoding=True
        )
        if label_dict is not None:
            label = label_dict[description][0]
            label = label.float()
        else:
            label = 1
        data = {"x": seq_embedding, "y": label, "bppm": pair_matrix}
        torch.save(data, file)

    def __len__(self):
        return len(self._files)

    def __getitem__(self, item):
        file = self._files[item]
        data = torch.load(file)
        x = data["x"]
        n = x.shape[0]
        e = x.shape[1]
        x_x = x.repeat(n, 1)
        x_y = x.repeat(1, n).reshape(-1, e)
        new = torch.cat((x_x, x_y), dim=1).reshape(n, n, -1)
        bppm = data["bppm"]
        pair_matrix = torch.cat((bppm, new), dim=-1)
        y = data["y"]
        mask = 1
        if not isinstance(y, int):
            if y.shape[0] < self.max_length:
                pad_val = self.max_length - y.shape[0]
                y = F.pad(y, (0, pad_val, 0, pad_val),
                                    "constant", 0)
            mask = (y > 0).float()
            # y = y.triu()
        return x, pair_matrix, y, mask


if __name__ == '__main__':
    rna_dataset = RNAPairDataset(
        data="Datasets/generation/random_40_200.fasta",
        label_dir="Datasets/generation/labels/new_random_s40_e200_tmp37/",
        dataset_path="/home/rabsch/Documents/foo/",
        num_threads=1,
        max_length=200
    )
    p = rna_dataset[2]
