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
from RNAdist.DPModels.viennarna_helpers import (
    fold_bppm, set_md_from_config, plfold_bppm)


NUCLEOTIDE_MAPPING = {
    "A": [1, 0, 0, 0],
    "U": [0, 1, 0, 0],
    "C": [0, 0, 1, 0],
    "G": [0, 0, 0, 1]
}


def pos_encoding(idx, dimension):
    enc = []
    for dim in range(int(dimension / 2)):
        w = 1 / (10000 ** (2 * dim / dimension))
        sin = math.sin(w * idx)
        cos = math.cos(w * idx)
        enc += [sin, cos]
    return enc


def positional_encode_seq(seq_embedding, pos_encoding_dim: int = 4):
    pos_enc = []
    for idx, letter in enumerate(seq_embedding):
        enc = pos_encoding(idx, pos_encoding_dim)
        pos_enc.append(enc)
    pos_enc = torch.tensor(pos_enc, dtype=torch.float)
    seq_embedding = torch.cat((seq_embedding, pos_enc), dim=1)
    return seq_embedding


def _pad_stuff(seq_embedding, pair_matrix, up_to):
    if seq_embedding.shape[0] < up_to:

        pad_val = up_to - seq_embedding.shape[0]
        seq_embedding = F.pad(seq_embedding, (0, 0, 0, pad_val),
                              "constant", 0)
        pair_matrix = F.pad(pair_matrix, (0, 0, 0, pad_val, 0, pad_val),
                            "constant", 0)
    return seq_embedding, pair_matrix


class RNADATA():
    def __init__(self, sequence, description=None, md=None):
        self.sequence = sequence.replace("T", "U")
        self.description = description
        self.md = md

    def to_tensor(self, max_length, positional_encoding, pos_encoding_dim=4, mode: str = "fold"):
        if mode == "fold":
            bppm = fold_bppm(self.sequence, self.md)
        elif mode == "plfold":
            assert self.md.max_bp_span > 0 and self.md.window_size > 0
            bppm = plfold_bppm(self.sequence, self.md.window_size, self.md.max_bp_span)
        else:
            raise ValueError(f"mode must be one of [fold, plfold] but is {mode}")
        bppm = torch.tensor(bppm, dtype=torch.float)
        bppm = self._add_upprob(bppm)
        seq_embedding = [NUCLEOTIDE_MAPPING[m][:] for m in self.sequence]
        seq_embedding = torch.tensor(seq_embedding, dtype=torch.float)
        if positional_encoding:
            seq_embedding = positional_encode_seq(
                seq_embedding, pos_encoding_dim
            )
        pair_matrix = bppm[:, :, None]
        seq_embedding, pair_matrix = _pad_stuff(
            seq_embedding, pair_matrix, max_length
        )
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


class RNADataset(Dataset):
    def __init__(self,
                 data: str,
                 label_dir: Union[str, os.PathLike, None],
                 dataset_path: str = "./",
                 num_threads: int = 1,
                 max_length: int = 200,
                 md_config=None
                 ):
        self.dataset_path = dataset_path
        self.data = data
        self.extension = "_data.pt"
        self.num_threads = num_threads
        self.max_length = max_length
        self.md_config = md_config if md_config is not None else {}
        self.label_dir = label_dir
        if self.label_dir is not None:
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
        if not os.path.exists(self.dataset_path):
            os.makedirs(dataset_path, exist_ok=True)

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

    @staticmethod
    def pair_rep_from_single(x):
        n = x.shape[0]
        e = x.shape[1]
        x_x = x.repeat(n, 1)
        x_y = x.repeat(1, n).reshape(-1, e)
        pair_rep = torch.cat((x_x, x_y), dim=1).reshape(n, n, -1)
        return pair_rep


class RNAPairDataset(RNADataset):
    def __init__(self,
                 data: str,
                 label_dir: Union[str, os.PathLike, None],
                 dataset_path: str = "./",
                 num_threads: int = 1,
                 max_length: int = 200,
                 md_config=None):
        super().__init__(
            data, label_dir, dataset_path, num_threads, max_length, md_config
        )

        if not self._dataset_generated(self._files):
            self.generate_dataset()

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
        pair_rep = self.pair_rep_from_single(x)
        bppm = data["bppm"]
        pair_matrix = torch.cat((bppm, pair_rep), dim=-1)
        y = data["y"]
        seq_len = len(self.rna_graphs[item][1])
        pad_val = self.max_length - seq_len
        if not isinstance(y, int):
            if y.shape[0] < self.max_length:
                y = F.pad(y, (0, pad_val, 0, pad_val),
                                    "constant", 0)
        mask = torch.zeros(self.max_length, self.max_length)
        mask[:seq_len, :seq_len] = 1
        return x, pair_matrix, y, mask, item


class RNAWindowDataset(RNADataset):

    def __init__(self, data: str,
                 label_dir: Union[str, os.PathLike, None],
                 dataset_path: str = "./",
                 num_threads: int = 1,
                 max_length: int = 200,
                 md_config=None,
                 step_size: int = 1
                 ):
        super().__init__(
            data, label_dir, dataset_path, num_threads, max_length, md_config)
        if self.label_dir is not None:
            assert "window_size" in self.md_config
            assert "max_bp_span" in self.md_config
        self.step_size = step_size
        if not self._dataset_generated(list(self.files.values())):
            self.generate_dataset()

    def seq_indices(self, seq):
        return range(
            0, len(seq) - self.max_length + self.step_size, self.step_size
        )

    @cached_property
    def rna_graphs(self):
        data = []
        descriptions = set()
        for seq_record in SeqIO.parse(self.data, "fasta"):
            assert seq_record.description not in descriptions, "Fasta headers must be unique"
            indices = self.seq_indices(seq_record.seq)
            indices = indices if indices else [0]
            data.append((seq_record.description, indices, str(seq_record.seq)))
            descriptions.add(seq_record.description)
        return data

    def generate_dataset(self):
        calls = []
        for idx, seq_data in enumerate(self.rna_graphs):
            file = self.files[seq_data[0]]
            call = [file, seq_data, self.max_length, self.md_config, self.label_dict]
            calls.append(call)
        if self.num_threads == 1:
            for call in calls:
                self.mp_create_wrapper(*call)
        else:
            with Pool(self.num_threads) as pool:
                pool.starmap(self.mp_create_wrapper, calls)

    @staticmethod
    def mp_create_wrapper(file, seq_data, max_length, md_config, label_dict):
        description, indices, seq = seq_data
        md = RNA.md()
        set_md_from_config(md, md_config)
        md.max_bp_span = max_length
        md.window_size = max_length
        rna_data = RNADATA(seq, description, md)
        bppm, seq_embedding = rna_data.to_tensor(
            max_length,
            positional_encoding=False,
            mode="plfold"
        )
        if label_dict is not None:
            label = label_dict[description][0]
            label = label.float()
        else:
            label = 1
        data = {"x": seq_embedding, "bppm": bppm, "y": label}
        torch.save(data, file)

    @cached_property
    def _files(self):
        actual_files = {}
        data_array = []
        idx_mapping = {}
        for desc, indices, seq_data in self.rna_graphs:
            file = os.path.join(
                self.dataset_path, f"{desc}_{self.extension}"
            )
            actual_files[desc] = file
            idx_mapping[desc] = indices
            for index in indices:
                data_array.append((file, index, desc))
        return data_array, actual_files, idx_mapping

    @cached_property
    def index_mapping(self):
        return self._files[2]

    @cached_property
    def data_array(self):
        return self._files[0]

    @cached_property
    def files(self):
        return self._files[1]

    def __len__(self):
        return len(self.data_array)

    def __getitem__(self, item):
        file, index, _ = self.data_array[item]
        data = torch.load(file)
        ml = self.max_length
        x = data["x"]
        x = x[index:index+ml]
        x = positional_encode_seq(
            x,
            pos_encoding_dim=4
        )
        pair_rep = self.pair_rep_from_single(x)
        bppm = data["bppm"]
        y = data["y"]
        sub_bppm = bppm[index:index + ml, index:index + ml, :]
        pair_matrix = torch.cat((sub_bppm, pair_rep), dim=-1)
        mask = torch.zeros(ml, ml)
        cur_len = x.shape[0]
        mask[:cur_len, :cur_len] = 1
        x, pair_matrix = _pad_stuff(seq_embedding=x,
                                    pair_matrix=pair_matrix,
                                    up_to=ml)
        if not isinstance(y, int):
            y = y[index:index + ml, index:index + ml]
            pad_val = ml - cur_len
            if y.shape[0] < self.max_length:
                y = F.pad(y, (0, pad_val, 0, pad_val),
                                    "constant", 0)
        return x, pair_matrix, mask, item, y


if __name__ == '__main__':
    rna_dataset = RNAPairDataset(
        data="Datasets/generation/random_40_200.fasta",
        label_dir="Datasets/generation/labels/new_random_s40_e200_tmp37/",
        dataset_path="/home/rabsch/Documents/foo/",
        num_threads=1,
        max_length=200
    )
    p = rna_dataset[2]
