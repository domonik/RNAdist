import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn import GINEConv, GATv2Conv
from typing import Iterable
import numpy as np

def batched_pair_rep_from_single(x):
    b, n, e = x.shape
    n = x.shape[1]
    e = x.shape[2]
    x_x = x.repeat(1, n, 1)
    x_y = x.repeat(1, 1, n).reshape(b, -1, e)
    pair_rep = torch.cat((x_x, x_y), dim=2).reshape(b, n, n, -1)
    return pair_rep

class TriangularUpdate(nn.Module):
    def __init__(self, embedding_dim, c=128, mode="in"):
        super().__init__()
        self.c = c
        assert mode in ["in", "out"]
        if mode == "in":
            self.equation = "bikc,bjkc->bijc"
        else:
            self.equation = "bkjc,bkic->bijc"
        self.embedding_dim = embedding_dim
        self.left_edges = nn.Linear(self.embedding_dim, self.c)
        self.right_edges = nn.Linear(self.embedding_dim, self.c)
        self.left_update = nn.Sequential(
            nn.Linear(self.embedding_dim, self.c),
            nn.SiLU()
        )
        self.right_update = nn.Sequential(
            nn.Linear(self.embedding_dim, self.c),
            nn.SiLU()
        )
        self.final_update = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.SiLU()
        )
        #torch.nn.init.constant_(self.final_update[0].bias.data, 1)
        self.rescale = nn.Linear(self.c, self.embedding_dim)
        self.e_norm = nn.LayerNorm(embedding_dim)
        self.c_norm = nn.LayerNorm(self.c)

    def forward(self, pair_rep, mask=None):
        pair_rep = self.e_norm(pair_rep)
        le = self.left_edges(pair_rep)
        re = self.right_edges(pair_rep)
        fu = self.final_update(pair_rep)
        if mask is not None:
            le = le * mask[..., None]
            re = re * mask[..., None]
        lu = self.left_update(pair_rep)
        ru = self.right_update(pair_rep)
        ru = ru * re
        lu = le * lu
        # shape will be [b, seq_len, seq_len, c] and wen want to sum over
        u = torch.einsum(self.equation, lu, ru)
        u = self.c_norm(u)
        u = self.rescale(u)
        u *= fu
        if mask is not None:
            u = u * mask[..., None]
        return u


class TriangularSelfAttention(nn.Module):
    def __init__(self, embedding_dim, c=32, heads=4, mode="in"):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.c = c
        assert mode in ["in", "out"]
        self.mode = mode
        self.heads = heads
        self.e_norm = nn.LayerNorm(self.embedding_dim)
        self.dpa_queries_lin = nn.Linear(self.embedding_dim, self.c * self.heads, bias=False)
        self.dpa_keys_lin = nn.Linear(self.embedding_dim, self.c * self.heads, bias=False)
        self.dpa_values_lin = nn.Linear(self.embedding_dim, self.c * self.heads, bias=False)
        self.pair_bias_lin = nn.Linear(self.embedding_dim, self.heads, bias=False)
        self.final_update = torch.nn.Sequential(
            nn.Linear(self.embedding_dim, self.c * self.heads),
            torch.nn.SiLU()
        )
        self.final_lin = torch.nn.Linear(self.heads * self.c, self.embedding_dim)

    def forward(self, pair_rep, mask=None):
        if self.mode == "out":
            pair_rep = torch.permute(pair_rep, (0, 2, 1, 3))
            if mask is not None:
                mask = torch.permute(mask, (0, 2, 1))
        pair_rep = self.e_norm(pair_rep)
        n, seq_len, _, _ = pair_rep.shape
        dpa_queries = self.dpa_queries_lin(pair_rep) * self.c ** (-0.5)
        dpa_queries = dpa_queries.reshape(n, seq_len, seq_len, self.heads, self.c)

        dpa_keys = self.dpa_keys_lin(pair_rep)
        dpa_keys = dpa_keys.reshape(n, seq_len, seq_len, self.heads, self.c)

        dpa_values = self.dpa_values_lin(pair_rep)
        dpa_values = dpa_values.reshape(n, seq_len, seq_len, self.heads, self.c)

        # shapes:
        # dpa_keys:         n,s,q,h,c
        # dpa_queries:      n,v,s,h,c
        # wanted:
        # dpa:              n,s,q,v,h
        # equation: "nsqhc,nsvhc->nsqvh"
        dpa = torch.einsum("nsqhc,nsvhc->nsqvh", dpa_keys, dpa_queries)
        pair_bias = self.pair_bias_lin(pair_rep)

        if mask is not None:
            bias = (1e9 * (mask - 1.))
            dpa = dpa + bias.unsqueeze(dim=1)[..., None] # TODO: check

        attention = F.softmax(dpa + pair_bias.unsqueeze(dim=1), dim=3)

        # shapes:
        # attention:        n,s,q,v,h
        # dpa_keys:         n,v,s,h,c
        # wanted:
        # out:              n,s,q,h,c
        # equation: "nsqvh,nsvhc->nsqhc"
        out = torch.einsum("nsqvh,nsvhc->nsqhc", attention, dpa_values)

        fu = self.final_update(pair_rep)
        fu = fu.reshape(n, seq_len, seq_len, self.heads, self.c)
        out *= fu
        out = out.reshape(n, seq_len, seq_len, self.heads * self.c)
        out = self.final_lin(out)
        if self.mode == "out":
            out = torch.permute(out, (0, 2, 1, 3))
        if mask is not None:
            out = out * mask[..., None]
        return out


class PairUpdate(nn.Module):
    def __init__(self, embedding_dim, fw: int = 4, checkpointing: bool = False, dropout: float = 0.2):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.fw = fw
        self.triangular_update_in = TriangularUpdate(self.embedding_dim)
        self.triangular_update_out = TriangularUpdate(
            self.embedding_dim,
            mode="out"
        )
        self.triangular_attention_in = TriangularSelfAttention(
            self.embedding_dim,
            mode="in"
        )
        self.triangular_attention_out = TriangularSelfAttention(
            self.embedding_dim,
            mode="out"
        )
        kernel = 5
        self.transition = nn.Sequential(
            nn.Conv2d(self.embedding_dim, self.fw * self.embedding_dim, kernel_size=kernel, padding=(kernel - 1) // 2),
            nn.SiLU(),
            nn.Conv2d(self.embedding_dim * self.fw, self.embedding_dim, kernel_size=kernel, padding=(kernel - 1) // 2)
        )
        self.forward = self.forward_cp if checkpointing else self.forward_no_cp
        self.dropout = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)
        self.dropout3 = nn.Dropout(p=dropout)
        self.dropout4 = nn.Dropout(p=dropout)
        self.dropout5 = nn.Dropout(p=dropout)

    def forward_no_cp(self, pair_rep, mask=None):
        #pair_rep = self.dropout(self.triangular_update_in(pair_rep, mask)) + pair_rep
        #pair_rep = self.dropout2(self.triangular_update_out(pair_rep, mask)) + pair_rep
        pair_rep = self.dropout3(self.triangular_attention_in(pair_rep, mask)) + pair_rep

        pair_rep = self.dropout4(self.triangular_attention_out(pair_rep, mask)) + pair_rep
        o_pair_rep = pair_rep
        pair_rep = pair_rep.permute(0, 3, 1, 2)

        pair_rep = self.dropout5(self.transition(pair_rep))
        pair_rep = pair_rep.permute(0, 2, 3, 1)
        pair_rep = pair_rep + o_pair_rep

        if mask is not None:
            pair_rep = pair_rep * mask[..., None]
        return pair_rep

    def forward_cp(self, pair_rep: torch.Tensor, mask: torch.Tensor = None):
        if not pair_rep.requires_grad:
            pair_rep.requires_grad = True
            if mask is not None:
                mask.requires_grad = True
        #pair_rep = self.dropout(checkpoint(self.triangular_update_in, pair_rep, mask, use_reentrant=False)) + pair_rep

        #pair_rep = self.dropout2(checkpoint(self.triangular_update_out, pair_rep, mask, use_reentrant=False)) + pair_rep

        pair_rep = self.dropout3(checkpoint(self.triangular_attention_in, pair_rep, mask, use_reentrant=False)) + pair_rep
        pair_rep = self.dropout4(checkpoint(self.triangular_attention_out, pair_rep, mask, use_reentrant=False)) + pair_rep
        o_pair_rep = pair_rep
        pair_rep = pair_rep.permute(0, 3, 1, 2)

        pair_rep = self.dropout5(self.transition(pair_rep))
        pair_rep = pair_rep.permute(0, 2, 3, 1)
        pair_rep = pair_rep + o_pair_rep

        if mask is not None:
            pair_rep = pair_rep * mask[..., None]
        return pair_rep


class PairUpdateSmall(nn.Module):
    def __init__(self, embedding_dim, fw: int = 4, checkpointing: bool = False):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.fw = fw
        self.triangular_update_in = TriangularUpdate(self.embedding_dim)
        self.triangular_update_out = TriangularUpdate(
            self.embedding_dim,
            mode="out"
        )
        self.transition = nn.Sequential(
            nn.Linear(self.embedding_dim, self.fw * self.embedding_dim),
            nn.SiLU(),
            nn.Linear(self.embedding_dim * self.fw, self.embedding_dim)
        )
        self.forward = self.forward_cp if checkpointing else self.forward_no_cp

    def forward_no_cp(self, pair_rep, mask=None):
        pair_rep = self.triangular_update_in(pair_rep, mask) + pair_rep
        pair_rep = self.triangular_update_out(pair_rep, mask) + pair_rep
        pair_rep = self.transition(pair_rep) + pair_rep
        return pair_rep

    def forward_cp(self, pair_rep, mask=None):
        if not pair_rep.requires_grad:
            pair_rep.requires_grad = True
            if mask is not None:
                mask.requires_grad = True
        pair_rep = checkpoint(self.triangular_update_in, pair_rep, mask, preserve_rng_state=False) + pair_rep
        pair_rep = checkpoint(self.triangular_update_out, pair_rep, mask, preserve_rng_state=False) + pair_rep
        pair_rep = self.transition(pair_rep) + pair_rep
        return pair_rep



class DISTAtteNCionESmall(nn.Module):
    def __init__(self, embedding_dim, nr_updates: int = 1, fw: int = 4, checkpointing: bool = False):
        super().__init__()
        self.nr_updates = nr_updates
        self.pair_updates = nn.ModuleList(
            PairUpdateSmall(embedding_dim, fw, checkpointing) for _ in range(self.nr_updates)
        )
        self.output = nn.Linear(embedding_dim, 1)

    def forward(self, batch, mask=None):
        pair_rep = batch["pair_rep"]
        for idx in range(self.nr_updates):
            pair_rep = self.pair_updates[idx](pair_rep, mask)
        out = self.output(pair_rep)
        out = torch.squeeze(out)
        out = torch.relu(out)
        if mask is not None:
            out = out * mask
        return out


class RNADISTAtteNCionE(nn.Module):
    def __init__(self, embedding_dim, nr_updates: int = 1, fw: int = 4, checkpointing: bool = False):
        super().__init__()
        self.nr_updates = nr_updates
        self.embedding = nn.Linear(embedding_dim, embedding_dim)
        self.init_norm = nn.LayerNorm(embedding_dim)
        self.pair_updates = nn.ModuleList(
            PairUpdate(embedding_dim, fw, checkpointing) for _ in range(self.nr_updates)
        )
        self.output = nn.Linear(embedding_dim, 1)
        self.layer_norm = nn.LayerNorm(embedding_dim)

    def forward(self, batch, mask=None):
        pair_rep = batch["pair_rep"]
        pair_rep = self.embedding(pair_rep)
        pair_rep = self.init_norm(pair_rep)
        for idx in range(self.nr_updates):
            pair_rep = self.pair_updates[idx](pair_rep, mask)
        pair_rep = self.layer_norm(pair_rep)
        out = self.output(pair_rep)
        out = torch.squeeze(out)
        out = torch.relu(out)
        if mask is not None:
                out = out * mask
        return out


class DISTAtteNCionEDual(nn.Module):
    def __init__(self, embedding_dim, fw: int = 4):
        super().__init__()
        self.nr_updates = 2

        self.pair_updates = nn.ModuleList(
            PairUpdate(embedding_dim, fw) for _ in range(self.nr_updates)
        )
        self.head1 = nn.Linear(embedding_dim, 1)
        self.head2 = nn.Linear(embedding_dim, 1)

    def forward(self, pair_rep, mask=None):
        head2_out = None
        for idx in range(self.nr_updates):
            pair_rep = self.pair_updates[idx](pair_rep, mask)
            if idx == 0:
                head2_out = self.head2(pair_rep)
        head2_out = torch.squeeze(head2_out)
        head2_out = torch.sigmoid(head2_out)
        out = self.head1(pair_rep)
        out = torch.squeeze(out)
        out = torch.sigmoid(out)
        if mask is not None:
            out = out * mask
        return out, head2_out


class CovarianceLoss(nn.Module):
    def __init__(self, reduction: str = "sum"):
        super().__init__()
        self.loss = nn.MSELoss(reduction=reduction)

    def forward(self, x, y):
        x = self.batch_cov(x)
        y = self.batch_cov(y)
        return self.loss(x, y)

    @staticmethod
    def batch_cov(points):
        B, N, D = points.size()
        mean = points.mean(dim=1).unsqueeze(1)
        diffs = (points - mean).reshape(B * N, D)
        prods = torch.bmm(diffs.unsqueeze(2), diffs.unsqueeze(1)).reshape(B, N,
                                                                          D, D)
        bcov = prods.sum(dim=1) / (N - 1)  # Unbiased estimate
        return bcov


class DiagOffsetLoss(nn.Module):
    def __init__(self, max_length, device):
        super().__init__()
        self.weights = torch.zeros(max_length, max_length, device=device)
        for x in range(max_length):
            i = torch.diag_embed(torch.ones(max_length-x, device=device) * (x+1), offset = x)
            self.weights += i
            if x != 0:
                self.weights += i.T

    def forward(self, x, y):
        return torch.sum(torch.square(x - y) * self.weights)


class WeightedDiagonalMSELoss(nn.Module):
    def __init__(self, alpha: float, device: str, offset: int = 0, reduction: str = "sum"):
        super().__init__()
        self.loss = nn.MSELoss(reduction=reduction)
        self.alpha = alpha
        self.offset = offset
        self.device = device
        self.__dummy = torch.tensor(1, device=device)
        if reduction == "sum":
            self.accum_fct = torch.sum
        elif reduction == "mean":
            self.accum_fct = torch.mean
        else:
            raise ValueError("No valid reduction")

    def forward(self, x, y, mask=None):
        size = y.shape[-1]
        weights = torch.zeros((size, size),  device=self.device)
        triu_indices = torch.triu_indices(size, size, offset=self.offset+1)
        # weights is the weights for non diagonal
        weights[triu_indices[0], triu_indices[1]] = self.alpha
        weights[triu_indices[1], triu_indices[0]] = self.alpha
        # weights2 is the weights for diagonal
        weights2 = torch.full((size, size), 1-self.alpha, device=self.device)
        weights2[triu_indices[0], triu_indices[1]] = 0
        weights2[triu_indices[1], triu_indices[0]] = 0
        if mask is not None:
            weights = weights * mask
            weights2 = weights2 * mask
        n_el_weights = torch.max(weights.count_nonzero(), self.__dummy)
        n_el_weights2 = torch.max(weights2.count_nonzero(), self.__dummy)
        loss = torch.sum((x - y) ** 2 * weights)
        loss = loss / n_el_weights
        loss2 = torch.sum((((x - y) ** 2) * weights2)) / n_el_weights2
        return loss + loss2


class GraphRNADISTAtteNCionE(nn.Module):
    def __init__(
            self,
            input_dim,
            embedding_dim,
            pair_dim,
            max_length,
            upper_bound: int,
            fw: int = 4,
            graph_layers: int = 4,
            dropout: float = 0.1,
            device: str = "cpu",
            checkpointing: bool = False,
            inference_batch_size: int = 10,
            inference: bool = False
    ):
        super().__init__()
        self.nr_updates = 2
        self.edge_dim = 2
        self.device = device
        self.graph_layers = graph_layers
        self.embedding_dim = embedding_dim
        self.input_dim = input_dim
        self.dropout = dropout
        self.pair_dim = pair_dim
        self.heads = 4
        self.inference_batch_size = inference_batch_size
        self.inference = inference


        self.graph_convolutions = nn.ModuleList(
                GATv2Conv(
                    in_channels=self.embedding_dim,
                    out_channels=self.embedding_dim,
                    heads=self.heads,
                    add_self_loops=False,
                    edge_dim=2
                )
                for idx, _ in enumerate(range(self.graph_layers))
        )
        self.graph_conv_adjust = nn.ModuleList(
            nn.Sequential(
                nn.Linear(self.heads * self.embedding_dim, self.embedding_dim),
                nn.LeakySiLU(),
                nn.LayerNorm(self.embedding_dim)
            )
                for idx, _ in enumerate(range(self.graph_layers))
        )

        self.pair_updates = nn.ModuleList(
            PairUpdate(self.embedding_dim * 2 + self.pair_dim, fw, checkpointing=checkpointing) for _
            in range(self.nr_updates)
        )
        self.input_lin = nn.Sequential(
            nn.Linear(self.input_dim, self.embedding_dim),
            nn.SiLU()
        )
        self.max_length = torch.tensor(max_length).to(self.device)
        self.upper_bound = torch.tensor(upper_bound).to(self.device)
        self.nodes_per_batch = self.upper_bound + self.max_length - 1

        self.intermediate_rescale = nn.Sequential(
            nn.Linear((self.graph_layers + 1) * self.embedding_dim, self.embedding_dim),
            nn.LeakySiLU(),
            nn.LayerNorm(self.embedding_dim),

        )
        self.out_conv = nn.Conv2d(
            in_channels=self.embedding_dim * 2 + self.pair_dim,
            out_channels=32,
            kernel_size=5,
            padding=2
        )
        self.out_norm = nn.LayerNorm(32)
        self.output = nn.Sequential(
            nn.Linear(32, 1),
            nn.SiLU()
        )

        self.graph_conv_function = self.graph_conv_checkpoint if checkpointing else self.graph_conv_wrapper
        self.forward = self.forward_inference if self.inference else self.forward_training
        self.pval = int((self.max_length - 1) / 2)
        if self.inference:
            self.weights = self.create_weight_tensor(int(self.max_length)).to(self.device)
        else:
            self.weights = 1

    @staticmethod
    def create_weight_tensor(n: int):
        t = np.zeros((n, n))
        for x in range(n):
            np.fill_diagonal(t[x:], np.full(shape=(n-x), fill_value=n-x))
        return torch.from_numpy(t + t.T - np.diag(np.diag(t)))


    def graph_conv_wrapper(self, x, edge_index, edge_attr):
        graph_embeddings = [x]
        for idx in range(self.graph_layers):
            x = self.graph_convolutions[idx](x, edge_index, edge_attr)
            x = self.graph_conv_adjust[idx](x)
            graph_embeddings.append(x)
        return graph_embeddings

    def graph_conv_checkpoint(self, x, edge_index, edge_attr):
        return checkpoint(self.graph_conv_wrapper, x, edge_index, edge_attr, preserve_rng_state=False)

    def forward_training(self, data, mask=None):
        x, edge_index, edge_attr, idx_info, bppm = data["x"], data["edge_index"], data["edge_attr"], data["idx_info"], data["pair_rep"]
        b = idx_info.shape[0]
        i, j = idx_info[:, 1], idx_info[:, 2]
        x = self.input_lin(x)
        graph_embeddings = self.graph_conv_function(x, edge_index, edge_attr)
        x = torch.concat(graph_embeddings, dim=1)

        # gets batch representation back and rescales to embedding_dim
        x = self.intermediate_rescale(x)
        x = torch.stack(torch.split(x, self.nodes_per_batch))

        # gets batched pair_representations
        pair_rep = batched_pair_rep_from_single(x)

        # this strange part cuts out a part of the pair_representation that is used for distance_calculation
        dummy = torch.empty(b, self.max_length, self.max_length, self.embedding_dim * 2, device=self.device)
        for idx in range(b):
            dummy[idx] = pair_rep[idx, i[idx]:i[idx]+self.max_length, j[idx]:j[idx]+self.max_length, :]
        pair_rep = dummy
        pair_rep = torch.concat((pair_rep, bppm), dim=-1)
        for idx in range(self.nr_updates):
            pair_rep = self.pair_updates[idx](pair_rep, mask)

        pair_rep = self.out_conv(pair_rep.permute(0, -1, 1, 2)).permute(0, 2, 3, 1)
        pair_rep = self.out_norm(torch.relu(pair_rep))
        out = self.output(pair_rep)
        out = torch.squeeze(out)
        if mask is not None:
            out = out * mask
        return out

    def full_inference(self, pair_rep, mask, slen):
        # now this is split into sub-batches
        out = []
        for i in range(0, slen, self.inference_batch_size):
            b = []
            m = []
            for j in range(i, min(i+self.inference_batch_size, slen)):
                b.append(pair_rep[0, j:j+self.max_length, j:j+self.max_length])
                m.append(mask[0, j:j+self.max_length, j:j+self.max_length])
            b = torch.stack(b)
            m = torch.stack(m)
            pr = b
            for idx in range(self.nr_updates):
                pr = self.pair_updates[idx](pr, m)

            pr = self.out_conv(pr.permute(0, -1, 1, 2)).permute(0, 2, 3, 1)
            pr = self.out_norm(torch.relu(pr))
            pr = self.output(pr)
            pr = torch.squeeze(pr) / self.weights
            if mask is not None:
                pr = pr * m
            out.append(pr.flatten())
        out = generate_output_tensor(out, self.max_length, slen, self.device)
        out = torch.sparse.sum(out, dim=2)
        return out.to_dense()[self.pval:self.pval+slen, self.pval:self.pval+slen]

    def sites_inference(self, pair_rep, mask, slen, sites):
        out = torch.zeros(slen, slen + 2*self.pval)

        for i in range(0, sites.shape[-1], self.inference_batch_size):
            b = []
            m = []
            si = []
            for j in range(i, min(i+self.inference_batch_size, sites.shape[-1])):
                s = sites[0, j]
                si.append(s)
                b.append(pair_rep[0, s:s+self.max_length, s:s+self.max_length])
                m.append(mask[0, s:s+self.max_length, s:s+self.max_length])
            b = torch.stack(b)
            m = torch.stack(m)
            pr = b
            for idx in range(self.nr_updates):
                pr = self.pair_updates[idx](pr, m)
            pr = self.out_conv(pr.permute(0, -1, 1, 2)).permute(0, 2, 3, 1)
            pr = self.out_norm(torch.relu(pr))
            pr = self.output(pr)
            pr = torch.squeeze(pr)
            if mask is not None:
                pr = pr * m
            for idx, entry in enumerate(si):

                ipr = pr[idx, self.pval, :]
                out[entry, entry:entry+self.max_length] = ipr
        out = out[:, self.pval:self.pval+slen]
        return out

    def forward_inference(self, data, mask=None):
        """ This forward function is called during inference time

        It expects a single sequence as input and will split the windows accordingly

        Args:
            data: The pytroch geometric data object which acts like a dictionary
            mask: a mask to apply
        Returns:

        """
        x, edge_index, edge_attr, idx_info, bppm = data["single_x"], data["edge_index"], data["edge_attr"], data["idx_info"], data["pair_rep"]
        x = x.squeeze()
        slen = idx_info.item()
        assert bppm.shape[0] == 1
        x = self.input_lin(x)
        graph_embeddings = self.graph_conv_function(x, edge_index, edge_attr)
        x = torch.concat(graph_embeddings, dim=1)
        x = self.intermediate_rescale(x)
        x = x.unsqueeze(0)
        pair_rep = batched_pair_rep_from_single(x)

        # make sure to use the whole bppm here
        pair_rep = torch.concat((pair_rep, bppm), dim=-1)
        if data["sites"].dtype == torch.bool:
            return self.full_inference(pair_rep, mask, slen)
        else:
            return self.sites_inference(pair_rep, mask, slen, data["sites"])


class GraphRNADIST(nn.Module):
    def __init__(
            self,
            input_dim,
            embedding_dim, pair_dim,
            graph_layers: int = 4,
            device: str = "cpu",
            checkpointing: bool = False
    ):
        super().__init__()
        self.edge_dim = 2
        self.device = device
        self.graph_layers = graph_layers
        self.embedding_dim = embedding_dim
        self.input_dim = input_dim
        self.pair_dim = pair_dim
        self.heads = 4
        self.token_range = range(3, 10, 2)
        self.input_convs = nn.ModuleList(
            nn.Conv1d(self.input_dim, self.embedding_dim, kernel_size=x, padding=int((x - 1) / 2)) for x in
            self.token_range
        )
        self.conv_rescale = nn.Sequential(
            nn.Linear(len(self.token_range) * self.embedding_dim, self.embedding_dim),
            nn.LeakySiLU(),
            nn.LayerNorm(self.embedding_dim),

        )
        self.graph_convolutions = nn.ModuleList(
            GATv2Conv(
                in_channels=self.embedding_dim,
                out_channels=self.embedding_dim,
                heads=self.heads,
                add_self_loops=False,
                edge_dim=2
            )
            for idx, _ in enumerate(range(self.graph_layers))
        )
        self.graph_conv_adjust = nn.ModuleList(
            nn.Sequential(
                nn.Linear(self.heads * self.embedding_dim, self.embedding_dim),
                nn.LeakySiLU(),
                nn.LayerNorm(self.embedding_dim)
            )
            for idx, _ in enumerate(range(self.graph_layers))
        )

        self.intermediate_rescale = nn.Sequential(
            nn.Linear((self.graph_layers + 1) * self.embedding_dim, self.embedding_dim),
            nn.LeakySiLU(),
            nn.LayerNorm(self.embedding_dim),

        )
        self.out_conv = nn.Conv2d(
            in_channels=self.embedding_dim * 2 + self.pair_dim,
            out_channels=32,
            kernel_size=5,
            padding=2
        )
        self.out_norm = nn.LayerNorm(32)
        self.output = nn.Sequential(
            nn.Linear(32, 1),
            nn.SiLU()
        )

        self.graph_conv_function = self.graph_conv_checkpoint if checkpointing else self.graph_conv_wrapper

    def graph_conv_wrapper(self, x, edge_index, edge_attr):
        graph_embeddings = [x]
        for idx in range(self.graph_layers):
            x = self.graph_convolutions[idx](x, edge_index, edge_attr)
            x = self.graph_conv_adjust[idx](x)
            graph_embeddings.append(x)
        return graph_embeddings

    def graph_conv_checkpoint(self, x, edge_index, edge_attr):
        return checkpoint(self.graph_conv_wrapper, x, edge_index, edge_attr, preserve_rng_state=False)

    def forward(self, data, mask):
        x, edge_index, edge_attr, idx_info, bppm = data["single_x"], data["edge_index"], data["edge_attr"], data["idx_info"], \
            data["pair_rep"]
        batch_size, upper_bound = bppm.shape[0:2]
        ix = []
        for idx in range(len(self.token_range)):
            ix.append(self.input_convs[idx](x.permute(0, 2, 1)).permute(0, 2, 1).squeeze())
        x = torch.concatenate(ix, dim=-1)
        x = self.conv_rescale(x)
        x = torch.reshape(x, (batch_size * upper_bound, self.embedding_dim))
        graph_embeddings = self.graph_conv_function(x, edge_index, edge_attr)
        x = torch.concat(graph_embeddings, dim=1)
        x = self.intermediate_rescale(x)
        x = torch.reshape(x, (batch_size, upper_bound, self.embedding_dim))
        pair_rep = batched_pair_rep_from_single(x)

        pair_rep = torch.concat((pair_rep, bppm), dim=-1)
        pair_rep = self.out_conv(pair_rep.permute(0, -1, 1, 2)).permute(0, 2, 3, 1)
        pair_rep = self.out_norm(torch.relu(pair_rep))
        out = self.output(pair_rep)
        out = torch.squeeze(out)
        out = mask * out
        return out


def generate_indices(kernel_size: int, length: int, device: str = "cpu"):
    range = torch.arange(0, kernel_size, device=device)
    first_dim = range.repeat(kernel_size * length)
    second_dim = range.repeat_interleave(kernel_size).repeat(length)
    third_dim = torch.arange(0, length, device=device).repeat_interleave(kernel_size * kernel_size)
    idx_tensor = torch.stack((first_dim, second_dim, third_dim))
    idx_tensor[0:2] += idx_tensor[2]
    return idx_tensor


def generate_output_tensor(tensorlist: Iterable[torch.tensor], kernel_size: int, length: int, device: str):
    indices = generate_indices(kernel_size, length, device)
    tensorlist = torch.concat(tensorlist)
    sparse_tensor = torch.sparse_coo_tensor(indices, values=tensorlist, size=tuple(indices[:, -1] + 1))
    return sparse_tensor







if __name__ == '__main__':
    import time
    kernel_size = 10
    length = 100
    indices = generate_indices(kernel_size, length)
    vecs = []
    for x in range(length):
        vecs.append((torch.ones(kernel_size, kernel_size, dtype=torch.float32) * x).flatten())
    sparse_tensor = generate_output_tensor(vecs, kernel_size, length, device="cpu")
    time.sleep(10)




