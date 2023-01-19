import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


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
            nn.Sigmoid()
        )
        self.right_update = nn.Sequential(
            nn.Linear(self.embedding_dim, self.c),
            nn.Sigmoid()
        )
        self.final_update = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
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
    def __init__(self, embedding_dim, c=64, heads=4, mode="in"):
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
            torch.nn.Sigmoid()
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
    def __init__(self, embedding_dim, fw: int = 4, checkpointing: bool = False):
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
        self.transition = nn.Sequential(
            nn.Linear(self.embedding_dim, self.fw * self.embedding_dim),
            nn.Sigmoid(),
            nn.Linear(self.embedding_dim * self.fw, self.embedding_dim)
        )
        self.forward = self.forward_cp if checkpointing else self.forward_no_cp

    def forward_no_cp(self, pair_rep, mask=None):
        pair_rep = self.triangular_update_in(pair_rep, mask) + pair_rep
        pair_rep = self.triangular_update_out(pair_rep, mask) + pair_rep
        pair_rep = self.triangular_attention_in(pair_rep, mask) + pair_rep
        pair_rep = self.triangular_attention_out(pair_rep, mask) + pair_rep
        pair_rep = self.transition(pair_rep) + pair_rep
        if mask is not None:
            pair_rep = pair_rep * mask[..., None]
        return pair_rep

    def forward_cp(self, pair_rep: torch.Tensor, mask: torch.Tensor = None):
        if not pair_rep.requires_grad:
            pair_rep.requires_grad = True
            if mask is not None:
                mask.requires_grad = True
        pair_rep = checkpoint(self.triangular_update_in, pair_rep, mask, use_reentrant=False) + pair_rep
        pair_rep = checkpoint(self.triangular_update_out, pair_rep, mask, use_reentrant=False) + pair_rep
        pair_rep = checkpoint(self.triangular_attention_in, pair_rep, mask, use_reentrant=False) + pair_rep
        pair_rep = checkpoint(self.triangular_attention_out, pair_rep, mask, use_reentrant=False) + pair_rep
        pair_rep = self.transition(pair_rep) + pair_rep
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
            nn.Sigmoid(),
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

    def forward(self, pair_rep, mask=None):
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
        self.pair_updates = nn.ModuleList(
            PairUpdate(embedding_dim, fw, checkpointing) for _ in range(self.nr_updates)
        )
        self.output = nn.Linear(embedding_dim, 1)

    def forward(self, pair_rep, mask=None):
        for idx in range(self.nr_updates):
            pair_rep = self.pair_updates[idx](pair_rep, mask)
        out = self.output(pair_rep)
        out = torch.squeeze(out)
        out = torch.relu(out)
        if mask is not None:
            out = out * mask
        return out


class DISTAtteNCionEDual(nn.Module):

    def __init__(self, embedding_dim, nr_updates: int = 1, fw: int = 4, checkpointing: bool = False):
        super().__init__()
        self.nr_updates = nr_updates
        self.mask_perc = 0.1
        self.nr_inital_updates = 1
        inner_dim = embedding_dim * 2
        edhalf = int(embedding_dim / 2) + 1
        ed4 = 4 * embedding_dim
        self.keys = nn.ModuleList(nn.Linear(edhalf if x == 0 else ed4, ed4) for x in range(self.nr_inital_updates))
        self.values = nn.ModuleList(nn.Linear(edhalf if x == 0 else ed4, ed4) for x in range(self.nr_inital_updates))
        self.queries = nn.ModuleList(nn.Linear(edhalf if x == 0 else ed4, ed4) for x in range(self.nr_inital_updates))
        self.mha = nn.ModuleList(torch.nn.MultiheadAttention(embedding_dim * 4, 4, batch_first=True) for x in range(self.nr_inital_updates))
        self.gates = nn.ModuleList(
            nn.Sequential(nn.Linear(edhalf if x == 0 else ed4, ed4), nn.Sigmoid()) for x in range(self.nr_inital_updates)
        )
        #assert self.nr_updates >= 2, "Needs to have multiple update layers for DualMode"
        self.pair_updates = nn.ModuleList(
            PairUpdate(inner_dim * 4, fw, checkpointing) for _ in range(self.nr_updates)
        )
        self.layer_norms = nn.ModuleList(
            nn.LayerNorm(ed4) for _ in range(self.nr_inital_updates)
        )
        self.output = nn.Linear(inner_dim * 4, 1)
        self.head2 = nn.Linear(inner_dim *  4, 1)
        self.masked_nt_lin = nn.Sequential(nn.Linear(inner_dim * 4, inner_dim), nn.Sigmoid(),
                                           nn.Linear(inner_dim, int(inner_dim / 2)), nn.Sigmoid(),
                                           nn.Linear(int(inner_dim / 2), 4), nn.ReLU())
        self.masked_loss = nn.CrossEntropyLoss(reduction="none")
        self.test = nn.Linear(inner_dim * 4, inner_dim * 4)
        self.resize = nn.Linear(embedding_dim + 1, inner_dim * 4)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear) :
            torch.nn.init.uniform_(m.weight, -1, 1)

    def mha_wrapper(self, x, q, k, v, mask):
        return self.mha[x](q, k, v, key_padding_mask = mask[:, :, 0])

    def forward(self, pair_rep, mask=None):
        y = torch.diagonal(pair_rep, dim1=2, dim2=1).permute(0, 2, 1)

        y = y[:, :,  0:9]

        if mask is not None:
            y_mask = torch.diagonal(mask, dim1=2, dim2=1)
            y_dropout, pair_rep_dropout = self.get_random_mask(y, y_mask)
            y_only_nt = y[:, :, 1:5]
            pair_rep_old = pair_rep
            pair_rep = torch.concat((pair_rep[:, :, :, 1:] * pair_rep_dropout.unsqueeze(-1), pair_rep[:, :, :, 0].unsqueeze(-1)), dim=-1)
            y = y * y_dropout.unsqueeze(-1)

        # for x in range(self.nr_inital_updates):
        #     k, v, q = self.keys[x](y), self.values[x](y), self.queries[x](y)
        #     y = checkpoint(self.mha_wrapper, x, q, k, v, mask, preserve_rng_state=True)[0] + self.gates[x](y)
        #     #y = self.layer_norms[x](y)
        #     #y, _ = self.mha(q, k, v, key_padding_mask = mask[:, :, 0])
        old_pr = self.resize(pair_rep_old)
        pair_rep = old_pr

        head2_out = None
        for idx in range(self.nr_updates):
            pair_rep = self.pair_updates[idx](pair_rep, mask ) + pair_rep
            # if idx == self.nr_updates - 2:
            #     head2_out = self.head2(pair_rep + old_pr)
        pair_rep = pair_rep
        #print(torch.histogram(old_pr, 10), old_pr.min(), old_pr.max())
        #print(torch.histogram(pair_rep, 10), pair_rep.min(), pair_rep.max())
        pair_rep = F.sigmoid(self.test(pair_rep))

        y = torch.sum(pair_rep, dim=-2)
        masked_y_pred = self.masked_nt_lin(y)
        l1 = self.masked_nt_pred(masked_y_pred, y_only_nt, y_dropout, y_mask)

        # head2_out = torch.squeeze(head2_out)
        # head2_out = torch.sigmoid(head2_out)
        out = self.output(pair_rep)
        out = torch.squeeze(out)
        out = torch.relu(out)
        if mask is not None:
            out = out * mask
            #head2_out = head2_out * mask
        return out, l1

    def get_random_mask(self, y, y_mask):
        y_dropout = (torch.rand((y.shape[0], y.shape[1]), device=y_mask.device) > self.mask_perc) * y_mask
        pair_rep_dropout = torch.bmm(y_dropout.unsqueeze(-1), y_dropout.unsqueeze(-1).permute(0, 2, 1))
        return y_dropout, pair_rep_dropout

    def masked_nt_pred(self, x, y, y_dropout, y_mask):
        y_drop = ~(y_dropout.type(torch.bool)) * y_mask
        loss = self.masked_loss(x.permute(0, -1, 1), y.permute(0, -1, 1))
        loss = loss * y_drop
        if torch.rand(size=(1,)) < 0.01:
            print((F.softmax(x, -1) * y_drop.unsqueeze(-1))[0][y_drop[0] != 0])
        loss = torch.sum(loss) / loss.count_nonzero()
        return loss

    def masked_bpp_pred(self, bpp_x, bpp_y, bpp_mask):
        pass




def batched_pair_rep_from_single(x):
    b, n, e = x.shape
    x_x = x.repeat(1, n, 1)
    x_y = x.repeat(1, 1, n).reshape(b, -1, e)
    pair_rep = torch.cat((x_x, x_y), dim=2).reshape(b, n, n, -1)
    return pair_rep

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




