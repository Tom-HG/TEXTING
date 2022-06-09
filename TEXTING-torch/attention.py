import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from math import sqrt

class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


class ProbMask():
    def __init__(self, B, H, L, index, scores, device="cpu"):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                    torch.arange(H)[None, :, None],
                    index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self):
        return self._mask

class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):  # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k))  # real U = U_part(factor*ln(L_k))*L_q
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   M_top, :]  # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else:  # use mask
            assert (L_Q == L_V)  # requires that L_Q == L_V, i.e. for self-attention only
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None],
        torch.arange(H)[None, :, None],
        index, :] = torch.matmul(attn, V)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) / L_V).double().to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.view(B, H, L_Q, -1)
        keys = keys.view(B, H, L_K, -1)
        values = values.view(B, H, L_K, -1)

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)

        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)

        # add scale factor
        scale = self.scale or 1. / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)

        return context.contiguous(), attn


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads=8, d_keys=None,
                 d_values=None, first=False):
        super(AttentionLayer, self).__init__()


        self.d_keys = d_keys or (d_model // n_heads)
        self.d_values = d_values or (d_model // n_heads)
        self.first = first

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, self.d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, self.d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, self.d_values * n_heads)
        self.out_projection = nn.Linear(self.d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, x, attn_mask):

        if self.first:
            queries = x[0]
            keys = x[1]
            values = x[2]

        else:
            queries = x
            keys = x
            values = x


        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


# import numpy as np
import torch
import torch.nn as nn
# import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features))).cuda()
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1))).cuda()
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

        self.norm = nn.LayerNorm(out_features)

    def forward(self, h, adj):
        Wh = torch.matmul(h, self.W)  # h.shape: (N, in_features), Wh.shape: (N, out_features) || h.shape: (B, N, in_features), Wh.shape: (B, N, out_features)
        # print(Wh.size())
        a_input = self._prepare_attentional_mechanism_input(Wh)  # (B,N,N,2*out_features)
        # print(a_input.size())
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))  # (B,N,N)
        # print(e.size())

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec) # (B,N,N)
        # print(attention.size())
        attention = F.softmax(attention, dim=2) # (B,N)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh) # (B,N) *(B, N, out_features)

        h_prime = self.norm(h_prime + Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size()[1]  # number of nodes

        # Below, two matrices are created that contain embeddings in their rows in different orders.
        # (e stands for embedding)
        # These are the rows of the first matrix (Wh_repeated_in_chunks):
        # e1, e1, ..., e1,            e2, e2, ..., e2,            ..., eN, eN, ..., eN
        # '-------------' -> N times  '-------------' -> N times       '-------------' -> N times
        #
        # These are the rows of the second matrix (Wh_repeated_alternating):
        # e1, e2, ..., eN, e1, e2, ..., eN, ..., e1, e2, ..., eN
        # '----------------------------------------------------' -> N times
        #

        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=1) #Wh.shape: (N, out_features)->(N * N, out_features) || (B, N, out_features)->(B, N * N, out_features)
        Wh_repeated_alternating = Wh.repeat(1, N, 1)
        # Wh_repeated_in_chunks.shape == Wh_repeated_alternating.shape == (N * N, out_features)

        # The all_combination_matrix, created below, will look like this (|| denotes concatenation):
        # e1 || e1
        # e1 || e2
        # e1 || e3
        # ...
        # e1 || eN
        # e2 || e1
        # e2 || e2
        # e2 || e3
        # ...
        # e2 || eN
        # ...
        # eN || e1
        # eN || e2
        # eN || e3
        # ...
        # eN || eN

        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=2)
        # all_combinations_matrix.shape == (B, N * N, 2 * out_features)

        return all_combinations_matrix.view(-1, N, N, 2 * self.out_features)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads, layers=1):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout
        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        # self.attn_layers = nn.ModuleList(GraphAttentionLayer(nhid * nheads, nhid * nheads, dropout=dropout, alpha=alpha, concat=False) for _ in range(layers))

        self.norm = nn.LayerNorm(nhid * nheads)
        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=2) #(B,N,nhid*nheads)
        x = self.norm(x)

        # for attn_layer in self.attn_layers:
        #     new_x = attn_layer(x, adj)
        #     x = self.norm(new_x+x)

        x = F.dropout(x, self.dropout, training=self.training)
        x = F.gelu(self.out_att(x, adj))
        # out = F.log_softmax(x, dim=2)

        return x


class GraphConvolution(nn.Module):
    def __init__(self, input_dim, \
                 output_dim, \
                 act_func=None, \
                 dropout_rate=0., \
                 bias=False):
        super(GraphConvolution, self).__init__()
        # self.support = support
        # self.featureless = featureless

        self.w = nn.Parameter(torch.randn(input_dim, output_dim)).cuda()
        # for i in range(len(self.support)):
        #     setattr(self, 'W{}'.format(i), nn.Parameter(torch.randn(input_dim, output_dim)))

        if bias:
            self.b = nn.Parameter(torch.zeros(1, output_dim)).cuda()

        self.act_func = act_func
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, support):
        # print(support)
        # d = torch.sum(support,dim=2)
        # D =
        # # print(d)
        # # D = torch.diag(d)
        # # print(D.size())
        # D = torch.inverse(D)**(1/2)
        # a = D.matmul(support).matmul(D)

        x = self.dropout(x)
        pre_sup = support.matmul(x)
        out = pre_sup.matmul(self.w)
        # for i in range(len(self.support)):
        #     if self.featureless:
        #         pre_sup = getattr(self, 'W{}'.format(i))
        #     else:
        #         pre_sup = x.mm(getattr(self, 'W{}'.format(i)))
        #
        #     if i == 0:
        #         out = self.support[i].mm(pre_sup)
        #     else:
        #         out += self.support[i].mm(pre_sup)

        if self.act_func is not None:
            out = self.act_func(out)

        self.embedding = out
        return out


class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim=200, dropout_rate=0., num_classes=2):
        super(GCN, self).__init__()

        # GraphConvolution
        self.layer1 = GraphConvolution(input_dim, hidden_dim, act_func=nn.GELU(),dropout_rate=dropout_rate)
        self.norm = nn.LayerNorm(hidden_dim)
        self.layer2 = GraphConvolution(hidden_dim, hidden_dim, dropout_rate=dropout_rate)

    def forward(self, x, support):
        out = self.layer1(x, support)
        out = self.norm(out)
        out = self.layer2(out, support)
        return out


if __name__ == "__main__":
    # layer = GraphAttentionLayer(in_features=300,out_features=8,dropout=0.5,alpha=0.2,concat=True)
    gat = GAT(nfeat=300,nhid=8,nclass=2,dropout=0.5,alpha=0.2,nheads=8).cuda()
    gcn = GCN(300,dropout_rate=0.5,num_classes=2)
    x = torch.rand(size=(1,3,300)).cuda()
    support = torch.randint(0,1,size=(1,3,3)).float().cuda()
    mask = torch.rand(size=(32,45,1)).cuda()
    y = torch.randint(0,1,(1,2)).cuda()
    out = gat(x,support)

    print(out.size())
    # print(out)