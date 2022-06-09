import torch
import torch.nn as nn
import numpy as np
from math import sqrt
from transformers import BertModel, BertConfig
from transformers import BertTokenizer, BertModel
class gru_unit(nn.Module):
    """GRU unit with 3D tensor inputs."""
    def __init__(self, output_dim, act=nn.ReLU, dropout_rate=0.5):
        super(gru_unit, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.act = act()
        self.z0 = nn.Linear(output_dim, output_dim)
        self.z1 = nn.Linear(output_dim, output_dim)
        self.r0 = nn.Linear(output_dim, output_dim)
        self.r1 = nn.Linear(output_dim, output_dim)
        self.h0 = nn.Linear(output_dim, output_dim)
        self.h1 = nn.Linear(output_dim, output_dim)
        self.init_params()
    def forward(self, support, x, mask):
        # message passing
        support = self.dropout(support)
        a = torch.matmul(support, x)

        # update gate
        z0 = self.z0(a)
        z1 = self.z1(x)
        z = (z0 + z1)
        z = torch.sigmoid(z)

        # reset gate
        r0 = self.r0(a)
        r1 = self.r1(x)
        r = (r0 + r1)
        r = torch.sigmoid(r)

        # update embeddings
        h0 = self.h0(a)
        h1 = self.h1(r*x)
        h = self.act(mask * (h0 + h1))

        return h * z + x * (1 - z)

    def init_params(self):
        nn.init.normal_(tensor=self.z0.weight, mean=0, std=0.1)
        nn.init.normal_(tensor=self.z0.bias, mean=0, std=0.1)
        nn.init.normal_(tensor=self.z1.weight, mean=0, std=0.1)
        nn.init.normal_(tensor=self.z1.bias, mean=0, std=0.1)
        nn.init.normal_(tensor=self.r0.weight, mean=0, std=0.1)
        nn.init.normal_(tensor=self.r0.bias, mean=0, std=0.1)
        nn.init.normal_(tensor=self.r1.weight, mean=0, std=0.1)
        nn.init.normal_(tensor=self.r1.bias, mean=0, std=0.1)
        nn.init.normal_(tensor=self.h0.weight, mean=0, std=0.1)
        nn.init.normal_(tensor=self.h0.bias, mean=0, std=0.1)
        nn.init.normal_(tensor=self.h1.weight, mean=0, std=0.1)
        nn.init.normal_(tensor=self.h1.bias, mean=0, std=0.1)

class GraphLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.5, act=nn.ELU, steps=2):
        super(GraphLayer, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        # self.support = support
        # self.mask = mask
        # self.dropout_rate = dropout
        self.dropout = nn.Dropout(dropout)
        self.act = act()
        self.steps = steps
        self.gru = gru_unit(output_dim, act, dropout)
        self.encode = nn.Linear(input_dim,output_dim)

    def forward(self, x, support, mask):
        # dropout
        x = self.dropout(x)

        # encode inputs
        x = self.encode(x) #[batch,len,input_dim] -> [batch,len,output_dim]
        output = mask * self.act(x) #[batch,len,output_dim]

        # convolve
        for _ in range(self.steps):
            output = self.gru(support, output, mask)

        return output

class ReadoutLayer(nn.Module):
    def __init__(self, input_dim, output_dim, act=nn.Tanh, dropout=0.5):
        super(ReadoutLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.act = act()
        self.dropout = nn.Dropout(dropout)

        # self.inter = nn.Linear(3,1)
        self.att = nn.Linear(input_dim,1)
        self.emb = nn.Linear(input_dim, input_dim)
        self.mlp = nn.Linear(input_dim, output_dim)

    def forward(self, x, support, mask):
        # soft attention
        att = torch.sigmoid(self.att(x)) #[batch,length,input_dim] - [batch,1,input_dim]
        emb = self.act(self.emb(x)) #[batch,length,input_dim] - [batch,length,input_dim]

        N = mask.sum(dim=1)
        M = (mask - 1) * 1e9

        # graph summation
        g = mask * att * emb
        M, idx = (g + M).max(dim=1)
        g = g.sum(dim=1) / N + M
        g = self.dropout(g)

        # classification
        output = self.mlp(g)

        return output

import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, d_model, n_heads=8, first=False):
        super(SelfAttention,self).__init__()
        self.d_keys = d_model // n_heads
        self.d_values = d_model // n_heads
        # These compute the queries, keys and values for all
        # heads (as a single concatenated vector)
        self.first = first
        self.tokeys    = nn.Linear(d_model, self.d_keys * n_heads, bias=False)
        self.toqueries = nn.Linear(d_model, self.d_keys * n_heads, bias=False)
        self.tovalues  = nn.Linear(d_model, self.d_values * n_heads, bias=False)

        # This unifies the outputs of the different heads into
        # a single k-vector
        self.unifyheads = nn.Linear(self.d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.init_params()

    def forward(self, x):
        if self.first:
            b, l, _ = x[0].size()
            h = self.n_heads
            k = self.d_keys

            queries = self.tokeys(x[2]).view(b, l, h, k)
            keys    = self.toqueries(x[1]).view(b, l, h, k)
            values  = self.tovalues(x[0]).view(b, l, h, k)
        else:
            b, l, _ = x.size()
            h = self.n_heads
            k = self.d_keys

            queries = self.tokeys(x).view(b, l, h, k)
            keys = self.toqueries(x).view(b, l, h, k)
            values = self.tovalues(x).view(b, l, h, k)

        # - fold heads into the batch dimension
        keys = keys.transpose(1, 2).contiguous().view(b * h, l, k)
        queries = queries.transpose(1, 2).contiguous().view(b * h, l, k)
        values = values.transpose(1, 2).contiguous().view(b * h, l, k)
        queries = queries / (k ** (1/4))
        keys    = keys / (k ** (1/4))

        # - get dot product of queries and keys, and scale
        dot = torch.bmm(queries, keys.transpose(1, 2))
        # - dot has size (b*h, t, t) containing raw weights

        dot = F.softmax(dot, dim=2)
        # - dot now contains row-wise normalized weights
        # apply the self attention to the values
        out = torch.bmm(dot, values).view(b, h, l, k)
        # swap h, t back, unify heads
        out = out.transpose(1, 2).contiguous().view(b, l, h * k)
        return self.unifyheads(out)

    def init_params(self):
        nn.init.normal_(tensor=self.tokeys.weight, mean=0, std=0.1)
        nn.init.normal_(tensor=self.toqueries.weight, mean=0, std=0.1)
        nn.init.normal_(tensor=self.tovalues.weight, mean=0, std=0.1)
        nn.init.normal_(tensor=self.unifyheads.weight, mean=0, std=0.1)
        nn.init.normal_(tensor=self.unifyheads.bias, mean=0, std=0.1)

class InterAttention(nn.Module):
    def __init__(self, d_model, n_heads=8, first=False):
        super(InterAttention,self).__init__()
        self.d_keys = d_model // n_heads
        self.d_values = d_model // n_heads
        # These compute the queries, keys and values for all
        # heads (as a single concatenated vector)
        self.first = first
        self.tokeys    = nn.Linear(d_model, self.d_keys * n_heads, bias=False)
        self.toqueries = nn.Linear(d_model, self.d_keys * n_heads, bias=False)
        self.tovalues  = nn.Linear(d_model, self.d_values * n_heads, bias=False)

        # This unifies the outputs of the different heads into
        # a single k-vector
        self.unifyheads = nn.Linear(self.d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.init_params()

    def forward(self, keys, queries, values):
        b, l, _ = keys.size()
        h = self.n_heads
        k = self.d_keys
        # - fold heads into the batch dimension
        keys = self.tokeys(keys)
        queries = self.toqueries(queries)
        values = self.tovalues(values)
        keys = keys.transpose(1, 2).contiguous().view(b * h, l, k)
        queries = queries.transpose(1, 2).contiguous().view(b * h, l, k)
        values = values.transpose(1, 2).contiguous().view(b * h, l, k)
        queries = queries / (k ** (1 / 4))
        values = values / (k ** (1 / 4))
        new_keys = keys / (k ** (1 / 4))

        # - get dot product of queries and keys, and scale
        dot_1 = torch.bmm(queries, new_keys.transpose(1, 2))
        dot_2 = torch.bmm(values, new_keys.transpose(1, 2))
        # - dot has size (b*h, t, t) containing raw weights

        dot_1 = F.softmax(dot_1, dim=2) /2
        dot_2 = F.softmax(dot_2, dim=2) /2

        # apply the self attention to the values
        out = torch.bmm(dot_1, keys).view(b, h, l, k) + torch.bmm(dot_2, keys).view(b, h, l, k)
        # swap h, t back, unify heads
        out = out.transpose(1, 2).contiguous().view(b, l, h * k)
        return self.unifyheads(out)

    def init_params(self):
        nn.init.normal_(tensor=self.tokeys.weight, mean=0, std=0.1)
        nn.init.normal_(tensor=self.toqueries.weight, mean=0, std=0.1)
        nn.init.normal_(tensor=self.tovalues.weight, mean=0, std=0.1)
        nn.init.normal_(tensor=self.unifyheads.weight, mean=0, std=0.1)
        nn.init.normal_(tensor=self.unifyheads.bias, mean=0, std=0.1)

from  attention import FullAttention, ProbAttention, AttentionLayer

class InformerEncoderLayer(nn.Module):
    def __init__(self, attention, d_model, attn='prob', d_ff=None, dropout=0.1, activation="relu"):
        super(InformerEncoderLayer,self).__init__()

        d_ff = d_ff or 4 * d_model
        Attn = ProbAttention if attn == 'prob' else FullAttention
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu() if activation == "relu" else F.gelu


    def forward(self, x, attn_mask=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn

from encoder import Encoder,EncoderLayer,ConvLayer
from decoder import DecoderLayer,Decoder
class Inter(nn.Module):
    def __init__(self, input_dim, hidden, output_dim, factor=5, n_heads=8, attn='prob', output_attention = False,
                 pooling='mean_and_max', activation="relu", sparse_inputs=False, dropout=0.1, depth=3, distil=True,num_layers=1):
        super(Inter, self).__init__()
        self.input_dim = input_dim
        self.hidden = hidden
        self.output_dim = output_dim
        self.activation = F.relu if activation == "relu" else F.gelu
        self.sparse_inputs = sparse_inputs
        self.dropout = nn.Dropout(dropout)

        # self.att = nn.Linear(input_dim,1)
        # self.emb = nn.Linear(input_dim, input_dim)

        # self.bn2d = nn.BatchNorm2d(num_features=input_dim)
        # self.propagation = nn.Linear(3,3,bias=False)
        self.pooling = pooling
        # self.bn = nn.BatchNorm1d(num_features=input_dim)
        self.output_layer = nn.Linear(input_dim, input_dim)
        self.mlp = nn.Linear(hidden, output_dim)

        self.d_model = input_dim
        self.d_ff = input_dim * 4
        self.selfatt = SelfAttention(input_dim, first=True)
        self.satt = SelfAttention(input_dim)
        Attn = ProbAttention if attn=='prob' else FullAttention
        d_model = input_dim
        self.att = AttentionLayer(attention=Attn(False, factor, attention_dropout=dropout, output_attention=output_attention),
                                d_model=d_model, n_heads=n_heads, first=True)
        # self.conv = ConvLayer(d_model)
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim*2)
        self.conv1 = nn.Conv1d(in_channels=self.d_model, out_channels=self.d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=self.d_ff, out_channels=self.d_model, kernel_size=1)
        self.elu = nn.ELU()
        self.depth = depth


        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=input_dim,
                            hidden_size=hidden,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=True)
        # tblocks = []
        # for i in range(depth):
        #     tblocks.append(InformerEncoderLayer(attention=self.att, d_model=input_dim))
        # self.tblocks = nn.Sequential(*tblocks)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(attention=Attn(False, factor, attention_dropout=dropout, output_attention=output_attention),
                                   d_model=d_model, n_heads=n_heads),
                    d_model,
                    self.d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(depth)
            ],
            [
                ConvLayer(
                    d_model
                ) for l in range(depth - 1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads),
                    d_model,
                    self.d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(depth)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        self.inatt = InterAttention(input_dim)
        self.softatt = nn.Linear(3,1)
        self.init_params()

    def forward(self, x, support, mask):

        #inter propagation
        # x = x * mask
        # new_x = torch.cat((x[0],x[1]),dim=2)
        # new_x = torch.cat((new_x,x[2]),dim=2)
        # print(new_x.size())
        # output = self.satt(new_x)

        output = self.selfatt(x) #[b,l,i]

        # new_x, att = self.att(x, mask)

        # output = torch.matmul(support, x)
        # output = output.sum(0)

        # y = new_x = self.norm1(new_x)


        # y = self.dropout(self.elu(self.conv1(y.transpose(-1, 1))))
        # y = self.dropout(self.conv2(y).transpose(-1, 1))
        # output = self.norm2(new_x + y)

        #transformers
        # x,_ = self.tblocks(output, att)
        # output, attns = self.encoder(output, attn_mask=mask)

        # output = self.decoder(output, output, x_mask=mask, cross_mask=mask)

        # h0, c0 = self.init_bilstm_state(output.size(0))
        # output = self.norm1(output)
        # output, (h_last, c_last) = self.lstm(output, (h0, c0))
        # output = self.norm2(output)
        # new_x = torch.zeros(size=x.size()).cuda()
        # new_x[0] = self.inatt(x[0], x[1], x[2])
        # new_x[1] = self.inatt(x[1], x[2], x[0])
        # new_x[2] = self.inatt(x[2], x[1], x[0])
        # output = self.selfatt(new_x)

        if self.pooling == "mean":
            output = torch.mean(output, dim=1)
        elif self.pooling == "max":
            output = torch.max(output, dim=1)[0]
        elif self.pooling == "mean_and_max":
            output = torch.mean(output, dim=1) + torch.max(output, dim=1)[0]
        elif self.pooling == "last":
            output = output[:, -1, :]
        else:
            output = output[:, 0, :]

        # output = self.bn(output)

        # output = self.dropout(output)

        # classification
        # output = self.activation(self.output_layer(output))
        logits = self.mlp(self.dropout(output))

        return logits

    def init_bilstm_state(self, batch_size):
        return (torch.zeros(self.num_layers*2, batch_size, self.hidden).cuda(),
                torch.zeros(self.num_layers*2, batch_size, self.hidden).cuda())

    def init_params(self):
        # nn.init.normal_(tensor=self.emb.weight, mean=0, std=0.1)
        # nn.init.normal_(tensor=self.emb.bias, mean=0, std=0.1)
        # nn.init.normal_(tensor=self.att.weight, mean=0, std=0.1)
        # nn.init.normal_(tensor=self.att.bias, mean=0, std=0.1)
        # nn.init.normal_(tensor=self.propagation.weight, mean=0, std=0.1)
        # nn.init.normal_(tensor=self.output_layer.weight, mean=0, std=0.1)
        # nn.init.normal_(tensor=self.output_layer.bias, mean=0, std=0.1)
        nn.init.normal_(tensor=self.mlp.weight, mean=0, std=0.1)
        nn.init.normal_(tensor=self.mlp.bias, mean=0, std=0.1)

class Model(nn.Module):
    def __init__(self,  input_dim, hidden, output_dim, dropout=0.5, steps=2):
        super(Model, self).__init__()
        self.input_dim = input_dim
        self.hidden = hidden
        self.output_dim = output_dim
        self.dropout = dropout
        self.GraphLayer = GraphLayer(input_dim=self.input_dim,
                                     output_dim=self.hidden,
                                     dropout=self.dropout,
                                     act=nn.Tanh,
                                     steps=steps)

        self.ReadoutLayer = ReadoutLayer(input_dim=self.hidden,
                                        output_dim=self.output_dim,
                                        act=nn.Tanh,
                                        dropout=self.dropout)

    def forward(self, x, support, mask):
        output = self.GraphLayer(x, support, mask)
        output = self.ReadoutLayer(output, support, mask)
        # output = torch.max(output, dim=0)[0]
        return output


class myModel(nn.Module):
    def __init__(self, input_dim, hidden, hidden_2, output_dim, dropout=0.5, steps=2, depth=2):
        super(myModel, self).__init__()
        self.input_dim = input_dim
        self.hidden = hidden
        self.hidden2 = hidden_2
        self.output_dim = output_dim
        self.dropout = dropout
        self.GraphLayer1 = GraphLayer(self.input_dim,
                                     self.hidden,
                                     self.dropout,
                                     act=nn.ELU,
                                     steps=steps)
        self.GraphLayer2 = GraphLayer(self.input_dim,
                                     self.hidden,
                                     self.dropout,
                                     act=nn.ELU,
                                     steps=steps)
        self.GraphLayer3 = GraphLayer(self.input_dim,
                                     self.hidden,
                                     self.dropout,
                                     act=nn.ELU,
                                     steps=steps)
        # self.l = nn.Linear(hidden, hidden_2)

        self.Inter = Inter(input_dim=self.hidden,
                           hidden = self.hidden2,
                           output_dim=self.output_dim,
                           pooling='mean_and_max',
                           activation="gelu",
                           sparse_inputs=False,
                           depth=depth)

        # self.ReadoutLayer = ReadoutLayer(
        #     input_dim=self.hidden,
        #     output_dim=self.output_dim,
        #     act=nn.Tanh,
        #     sparse_inputs=False,
        #     dropout=self.dropout)

    def forward(self, x, support, mask):
        output = torch.zeros(size=(3, x.size(0), x.size(1), self.hidden)).cuda()
        output[0] = self.GraphLayer1(x, support[0], mask)
        output[1] = self.GraphLayer2(x, support[1], mask)
        output[2] = self.GraphLayer3(x, support[2], mask)

        # output = self.l(output)
        output = self.Inter(output, support, mask)
        # print(output.size())

        # output = self.ReadoutLayer(output, support, mask)
        return output

class bertModel(nn.Module):
    def __init__(self,  bert_path, input_dim, hidden, output_dim, dropout=0.5, steps=2):
        super(bertModel, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name_or_path=bert_path)
        self.input_dim = input_dim
        self.hidden = hidden
        self.output_dim = output_dim
        self.dropout = dropout
        self.GraphLayer = GraphLayer(input_dim=self.input_dim,
                                     output_dim=self.hidden,
                                     dropout=self.dropout,
                                     act=nn.Tanh,
                                     steps=steps)

        self.ReadoutLayer = ReadoutLayer(input_dim=self.hidden,
                                        output_dim=self.output_dim,
                                        act=nn.Tanh,
                                        dropout=self.dropout)

    def forward(self, x, support, mask,attention_mask, sequence_length):
        x, _ = self.bert(input_ids=x, attention_mask=attention_mask)
        output = self.GraphLayer(x, support, mask)
        output = self.ReadoutLayer(output, support, mask)
        # output = torch.max(output, dim=0)[0]
        return output

from attention import GAT
class GATmodel(nn.Module):
    def __init__(self,  input_dim, hidden, output_dim, dropout=0.5):
        super(GATmodel, self).__init__()
        self.input_dim = input_dim
        self.hidden = hidden
        self.output_dim = output_dim
        self.dropout = dropout
        self.nhead = 8
        self.nhid = hidden // self.nhead
        self.pooling = "mean_and_max"
        self.gat = GAT(nfeat=input_dim,nhid=self.nhid,nclass=output_dim,dropout=dropout,alpha=0.02,nheads=8)
        self.norm = nn.LayerNorm(output_dim)


    def forward(self, x, support, mask):
        output = self.gat(x, support)
        output = self.norm(output)
        # print(output.size())

        if self.pooling == "mean":
            output = torch.mean(output, dim=1)
        elif self.pooling == "max":
            output = torch.max(output, dim=1)[0]
        elif self.pooling == "mean_and_max":
            output = torch.mean(output, dim=1) + torch.max(output, dim=1)[0]
        elif self.pooling == "last":
            output = output[:, -1, :]
        else:
            output = output[:, 0, :]
        return output

from attention import GCN
class GCNmodel(nn.Module):
    def __init__(self,  input_dim, hidden, output_dim, dropout=0.5):
        super(GCNmodel, self).__init__()
        self.input_dim = input_dim
        self.hidden = hidden
        self.output_dim = output_dim
        self.dropout = dropout
        self.pooling = "mean_and_max"
        self.gcn = GCN(input_dim=input_dim,hidden_dim=hidden,dropout_rate=dropout,num_classes=output_dim)

    def forward(self, x, support, mask):
        output = self.gcn(x, support)
        output = self.norm(output)
        # print(output.size())

        if self.pooling == "mean":
            output = torch.mean(output, dim=1)
        elif self.pooling == "max":
            output = torch.max(output, dim=1)[0]
        elif self.pooling == "mean_and_max":
            output = torch.mean(output, dim=1) + torch.max(output, dim=1)[0]
        elif self.pooling == "last":
            output = output[:, -1, :]
        else:
            output = output[:, 0, :]
        return output

class Gmodel(nn.Module):
    def __init__(self,  input_dim, hidden, output_dim, dropout=0.5):
        super(Gmodel, self).__init__()
        self.input_dim = input_dim
        self.hidden = hidden
        self.output_dim = output_dim
        self.dropout = dropout
        self.pooling = "mean_and_max"
        self.gcn = GCN(input_dim=input_dim,hidden_dim=hidden,dropout_rate=dropout,num_classes=output_dim)
        self.norm1 = nn.LayerNorm(hidden)

        self.nhead = 8
        self.nhid = hidden // self.nhead
        self.pooling = "mean_and_max"
        self.gat = GAT(nfeat=hidden, nhid=self.nhid, nclass=output_dim, dropout=dropout, alpha=0.02, nheads=8)
        # self.norm2 = nn.LayerNorm(output_dim)


    def forward(self, x, support, mask):
        output = self.gcn(x, support)
        output = self.norm1(output)

        output = self.gat(output, support)
        # output = self.norm2(output)
        # print(output.size())

        if self.pooling == "mean":
            output = torch.mean(output, dim=1)
        elif self.pooling == "max":
            output = torch.max(output, dim=1)[0]
        elif self.pooling == "mean_and_max":
            output = torch.mean(output, dim=1) + torch.max(output, dim=1)[0]
        elif self.pooling == "last":
            output = output[:, -1, :]
        else:
            output = output[:, 0, :]

        return output

class Bert(nn.Module):
    def __init__(self, name):
        super(Bert, self).__init__()
        self.bert = BertModel.from_pretrained(name)

    def forward(self,input_ids, token_type_ids, attention_mask):
        outputs,h = self.bert(input_ids, token_type_ids, attention_mask)

        return outputs

class Bert_Model(nn.Module):
    def __init__(self, input_dim, hidden, output_dim, dropout=0.5, steps=2):
        super(Bert_Model, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.model = Model(input_dim, hidden, output_dim, dropout=0.5, steps=2)
        self.linear = nn.Linear(input_dim,output_dim)

    def forward(self,input_ids, token_type_ids, attention_mask,support, mask):
        x,h = self.bert(input_ids, token_type_ids, attention_mask)
        outputs = self.model(x, support, mask)
        # outputs = self.linear(h).squeeze()
        
        return outputs


if __name__ == "__main__":
    # graphLayer = GraphLayer(300,64)
    # readoutLayer = ReadoutLayer(64,2)
    # model = Gmodel(300,64,2,0.5).cuda()
    # x = torch.rand(size=(32,45,300)).cuda()
    # support = torch.rand(size=(32,45,45)).cuda()
    # mask = torch.rand(size=(32,45,1)).cuda()
    # y = torch.randint(0,1,(32,2)).cuda()
    # # output = graphLayer(x, support, mask) #[32,20,128]
    # # output = readoutLayer(output, support, mask) #[32,2]
    # output = model(x, support, mask)
    # print(output)
    # print(output.size())
    # print(model)
    # configuration = BertConfig()
    # model = BertModel(configuration)
    # configuration = model.config
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # model = BertModel.from_pretrained('bert-base-uncased')
    inputs = tokenizer("harmless fun", return_tensors="pt")
    input_ids = inputs['input_ids']
    token_type_ids = inputs['token_type_ids']
    attention_mask = inputs['attention_mask']
    print(inputs)
    # print(inputs)
    # outputs,h = model(**inputs)
    model = Bert('bert-base-uncased')
    outputs = model(input_ids, token_type_ids, attention_mask)
    print(outputs.size())
    # last_hidden_states = outputs.last_hidden_state