import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import os
from RNAdist.NNModels.training_set_generation import LabelDict
from RNAdist.NNModels.nn_helpers import _scatter_triu_indices
from functools import cached_property
from typing import List, Union
from Bio import SeqIO
from torch.multiprocessing import Pool
import RNA
import itertools
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


def _pad_end(seq_embedding, pair_matrix, up_to, mask=None):
    if seq_embedding.shape[0] < up_to:

        pad_val = up_to - seq_embedding.shape[0]
        seq_embedding = F.pad(seq_embedding, (0, 0, 0, pad_val),
                              "constant", 0)
        pair_matrix = F.pad(pair_matrix, (0, 0, 0, pad_val, 0, pad_val),
                            "constant", 0)
        if mask is not None:
            mask = F.pad(mask, (0, pad_val, 0, pad_val),
                                "constant", 0)
            return seq_embedding, pair_matrix, mask
    if mask is not None:
        return seq_embedding, pair_matrix, mask
    return seq_embedding, pair_matrix


class RNADATA():
    def __init__(self, sequence, description=None, md=None):
        self.sequence = str(sequence).replace("T", "U")
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
        if mode == "fold":
            seq_embedding, pair_matrix = _pad_end(
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
                 max_length: int = 201,
                 md_config=None,
                 step_size: int = 1
                 ):
        super().__init__(
            data, label_dir, dataset_path, num_threads, max_length, md_config)
        self.step_size = step_size
        if self.step_size != 1:
            raise NotImplementedError("Step Size > 1 is not implemented yet")
        if not self.max_length % 2:
            raise ValueError(f"max_length must be uneven for window mode")
        if not self._dataset_generated([file for file in self.files]):
            self.generate_dataset()

    @cached_property
    def _seq_data(self):

        indices2seq = []
        files = []
        sequences = []
        total_len = 0
        for file_index, seq_record in enumerate(SeqIO.parse(self.data, "fasta")):
            desc = seq_record.description
            seq = str(seq_record.seq)
            seq_len = len(seq)
            start = total_len
            m = math.ceil(seq_len / self.step_size)
            matrix_elements = int(((m * m) - m) / 2 + m)
            total_len += matrix_elements
            end = total_len
            file = os.path.join(
                self.dataset_path, f"{desc}_{self.extension}"
            )
            files.append(file)
            sequences.append((desc, seq))
            indices2seq.append((file_index, start, end))
        return indices2seq, files, sequences

    @property
    def files(self):
        return self._seq_data[1]

    @property
    def index_to_data(self):
        return self._seq_data[0]

    @property
    def seq_data(self):
        return self._seq_data[2]


    def generate_dataset(self):
        calls = []
        for idx, file in enumerate(self.files):
                desc, seq = self.seq_data[idx]
                calls.append((seq, desc, file, self.label_dict, self.md_config))
        if self.num_threads == 1:
            for call in calls:
                self.mp_create_wrapper(*call)
        else:
            with Pool(self.num_threads) as pool:
                pool.starmap(self.mp_create_wrapper, calls)

    @staticmethod
    def mp_create_wrapper(seq, desc, file, label_dict, md_config):
        md = RNA.md()
        set_md_from_config(md, md_config)
        rna_data = RNADATA(seq, desc, md)
        bppm, seq_embedding = rna_data.to_tensor(
            max_length=len(seq),
            positional_encoding=False,
            mode="fold"
        )
        if label_dict is not None:
            label = label_dict[desc][0]
            label = label.float()
        else:
            label = 1
        data = {"x": seq_embedding, "bppm": bppm, "y": label}
        torch.save(data, file)

    def get_indes_from_ranges(self, index):
        right = len(self.index_to_data)
        left = 0
        while left <= right:
            file_idx = math.floor((left + right) / 2)
            file, lower, upper = self.index_to_data[file_idx]
            if index < lower:
                right = file_idx - 1
            elif index >= upper:
                left = file_idx + 1
            else:
                inner_index = index - lower
                return file, inner_index
        raise IndexError("File index out of range")


    def __getitem__(self, item):
        with torch.no_grad():
            file_index, inner_index = self.get_indes_from_ranges(item)
            data = torch.load(self.files[file_index])
            x = data["x"]
            y = data["y"]
            slen = x.shape[0]
            i, j = _scatter_triu_indices(slen, self.step_size)[inner_index, :]
            #start = int(torch.randint(0, 600, size=(1,)))
            start = 0
            positions = torch.tensor([pos_encoding(idx, 4) for idx in range(start,start+ x.shape[0])])
            x = torch.cat((x, positions), dim=1)
            bppm = data["bppm"]
            mask = torch.ones(*bppm.shape[0:2])
            pad_val = int((self.max_length - 1) / 2)
            x = F.pad(x,  (0, 0, pad_val, pad_val))
            pair_rep = self.pair_rep_from_single(x)
            pair_rep = pair_rep[i:i+self.max_length, j:j+self.max_length]
            bppm = F.pad(bppm,  (0, 0, pad_val, pad_val, pad_val, pad_val))
            bppm = bppm[i:i+self.max_length, j:j+self.max_length]
            mask = F.pad(mask, (pad_val, pad_val, pad_val, pad_val))
            mask = mask[i:i+self.max_length, j:j+self.max_length]
            pair_rep = torch.cat((bppm, pair_rep), dim=-1)
            if not isinstance(y, int):
                y = F.pad(y, (pad_val, pad_val, pad_val, pad_val))
                y = y[i:i + self.max_length, j:j + self.max_length]
            x = torch.zeros(self.max_length, 1)
            file_index = torch.tensor(file_index)
            idx_information = torch.stack((file_index, i, j), dim=0)
        return x, pair_rep, y, mask, idx_information

    def __len__(self):
        return self.index_to_data[-1][-1]




