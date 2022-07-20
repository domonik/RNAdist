import torch
import torch.nn as nn
import torch.nn.functional as F


class SequenceMultihead(torch.nn.Module):
    def __init__(self, embed_size, heads, head_dim):
        super().__init__()
        self.embed_size = embed_size
        self.heads = heads  # := c in alpha
        self.head_dim = head_dim

        self.values = nn.Linear(self.embed_size, self.head_dim * heads)
        self.keys = nn.Linear(self.embed_size, self.head_dim * heads)
        self.queries = nn.Linear(self.embed_size, self.head_dim * heads)

    def forward(self, values, keys, query):
        N = query.shape[0]

        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
        values = self.values(values)
        keys = self.keys(keys)
        query = self.queries(query)
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        query = query.reshape(N, query_len, self.heads, self.head_dim)
        dpa = torch.einsum("nqhc,nvhc->nqvh", [query, keys])
        return values, dpa


class SequencePairAttention(torch.nn.Module):
    def __init__(self, embed_size, pair_dim, heads, head_dim):
        super().__init__()
        self.seq_multihead = SequenceMultihead(embed_size, heads, head_dim)
        self.pair_dim = pair_dim
        self.head_dim = head_dim
        self.heads = heads
        self.embed_size = embed_size


        self.value_transform = nn.Linear(self.head_dim * self.heads, self.embed_size)
        self.pair_transform = nn.Linear(self.pair_dim, self.heads)
        self.pair_to_embed = nn.Linear(self.pair_dim * self.heads, self.embed_size)
        self.pair_to_pair = nn.Linear(self.pair_dim * self.heads, self.pair_dim)

    def forward(self, embedding, pair_rep):
        N = embedding.shape[0]
        seq_len = embedding.shape[1]
        values, dpa = self.seq_multihead(embedding, embedding, embedding)
        pair_bias = self.pair_transform(pair_rep)
        attention_weights = torch.softmax(pair_bias + dpa, dim=2)

        # attention.shape = (N, seq_len (q), seq_len (v), heads (h))
        # values.shape = (N, seq_len (v), heads (h), head_dim (c))
        # out.shape (N, seq_len (q), heads (h), head_dim(c))
        out = torch.einsum("nqvh,nvhc->nqhc", [attention_weights, values])
        out = out.reshape(N, seq_len, self.heads * self.head_dim)
        out = self.value_transform(out)

        # attention.shape = (N, seq_len (q), seq_len (v), heads (h))
        # pair_rep.shape = (N, seq_len (v), seq_len (q), pair_dim (cz))
        # out.shape (N, seq_len, heads, pair_dim)
        pair_out = torch.einsum("nqvh,nvqc->nqhc", [attention_weights, pair_rep])
        pair_out = pair_out.reshape(N, seq_len, self.heads * self.pair_dim)
        pair_add = self.pair_to_embed(pair_out)
        pair_out = self.pair_to_pair(pair_out)
        # pair_out.shape = (N, seq_len, pair_dim)
        # pair_out.shape expected = (N, seq_len, seq_len, pair_dim)
        pair_out = torch.einsum("biz,bjz->bijz", pair_out, pair_out)

        out = out + pair_add

        return out, pair_out


class DISTAtteNCionE(nn.Module):
    def __init__(self, embed_size, pair_dim, heads, head_dim, nr_attentions: int = 2):
        super().__init__()
        self.nr_attentions = nr_attentions
        self.attentions = nn.ModuleList(
            SequencePairAttention(
                embed_size,
                pair_dim,
                heads,
                head_dim
            ) for _ in range(nr_attentions)
        )
        self.convs = RNADistConvolutions(
            input_dim=embed_size,
            window_size=80,
            nr_convolutions=3,
            conv_dim=10,
            conv_out_dim=1,
            kernel_size=9,
            activation_fct=torch.nn.Sigmoid
        )

    def forward(self, x, bppm):
        for idx in range(self.nr_attentions):
            nx, _ = self.attentions[idx](x, bppm)
            if idx < self.nr_attentions - 1:
                nx = F.sigmoid(nx)
            x = nx + x
        # x.shape = (N, seq_len, embedding_dim)
        # out.shape = (N, seq_len, seq_len)
        x = torch.einsum("biz,bjz->bij", x, x)
        #x = torch.permute(x, (0, 3, 1, 2))
        #x = self.convs(x)
        x = F.relu(x)
        return x


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
    def __init__(self, embedding_dim, fw: int = 4):
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
            nn.ReLU(),
            nn.Linear(self.embedding_dim * self.fw, self.embedding_dim)
        )

    def forward(self, pair_rep, mask=None):
        pair_rep = self.triangular_update_in(pair_rep, mask) + pair_rep
        pair_rep = self.triangular_update_out(pair_rep, mask) + pair_rep
        pair_rep = self.triangular_attention_in(pair_rep, mask) + pair_rep
        pair_rep = self.triangular_attention_out(pair_rep, mask) + pair_rep
        pair_rep = self.transition(pair_rep) + pair_rep
        if mask is not None:
            pair_rep = pair_rep * mask[..., None]
        return pair_rep


class PairUpdateSmall(nn.Module):
    def __init__(self, embedding_dim, fw: int = 4):
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
            nn.ReLU(),
            nn.Linear(self.embedding_dim * self.fw, self.embedding_dim)
        )

    def forward(self, pair_rep, mask=None):
        pair_rep = self.triangular_update_in(pair_rep, mask) + pair_rep
        pair_rep = self.triangular_update_out(pair_rep, mask) + pair_rep
        pair_rep = self.transition(pair_rep) + pair_rep
        return pair_rep


class DISTAtteNCionESmall(nn.Module):
    def __init__(self, embedding_dim, nr_updates: int = 1, fw: int = 4):
        super().__init__()
        self.nr_updates = nr_updates
        self.pair_updates = nn.ModuleList(
            PairUpdateSmall(embedding_dim, fw) for _ in range(self.nr_updates)
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


class DISTAtteNCionE2(nn.Module):
    def __init__(self, embedding_dim, nr_updates: int = 1, fw: int = 4):
        super().__init__()
        self.nr_updates = nr_updates
        self.pair_updates = nn.ModuleList(
            PairUpdate(embedding_dim, fw) for _ in range(self.nr_updates)
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



if __name__ == '__main__':
    seq = "AAUGUGAACA" * 20
    m = [GraphDistance.NUCLEOTIDE_MAPPING[m] for m in seq]
    embedding = torch.tensor(m, dtype=torch.float)
    bppm = GraphDistance.fold_bppm(seq)
    bppm = bppm[:, :, None]
    bppm = bppm[None, :]
    bppm = bppm.repeat(20, 1, 1, 1)
    embedding = embedding[None, :]
    #model = SequencePairAttention(4, 1, 4, 64)
    model = PairUpdate(bppm.shape[-1])
    bppm = bppm.to("cuda")
    model.to("cuda")

    pred = model(bppm)
    input()
    p = 0





