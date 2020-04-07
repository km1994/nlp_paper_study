import math
import random

import numpy as np

import torch
import torch.nn as nn

import torch.nn.functional as F

from apex.normalization.fused_layer_norm import FusedLayerNorm as LayerNorm

import torch.utils
import torch.utils.checkpoint
tcheckpoint = torch.utils.checkpoint.checkpoint
#checkpoint = torch.utils.checkpoint.checkpoint
checkpoint = lambda f, *args, **kwargs: f(*args, **kwargs)

def attention(query, key, value, attn_mask=None, need_weights=True, dropout=None):
    # https://pytorchnlp.readthedocs.io/en/latest/_modules/torchnlp/nn/attention.html
    # Needs [batch, heads, seqlen, hid]

    batch_size, heads, query_len, dim = query.size()
    key_len = key.size(2)

    # Scaling by dim due to http://nlp.seas.harvard.edu/2018/04/03/attention.html
    attention_scores = torch.matmul(query, key.transpose(-1, -2).contiguous()) / math.sqrt(dim)
    if attn_mask is not None:
        attn_mask = attn_mask.view(1, 1, *attn_mask.shape[-2:])
        attention_scores = attention_scores + attn_mask # Mask is additive and contains -Infs

    attention_weights = F.softmax(attention_scores, dim=-1)
    if dropout:
        attention_weights = dropout(attention_weights)
    attention_weights = attention_weights.view(batch_size, heads, query_len, key_len)

    mix = torch.matmul(attention_weights, value)
    return mix, attention_weights

class Overparam(nn.Module):
    def __init__(self, nhid):
        super().__init__()
        self.l1 = nn.Linear(nhid, 2 * nhid)
        #self.l2 = nn.Linear(2 * nhid, 2 * nhid)
        self.inner_act = torch.tanh # GELU()
        self.nhid = nhid

    def forward(self, x):
        c, f = self.l1(x).split(self.nhid, dim=-1)
        #c, f = self.l2(self.inner_act(self.l1(x))).split(self.nhid, dim=-1)
        return torch.sigmoid(f) * torch.tanh(c)

class Attention(nn.Module):
    def __init__(self, nhid, q=True, k=False, v=False, r=False, heads=1, dropout=None):
        super().__init__()
        self.qs = nn.Parameter(torch.zeros(size=(1, 1, nhid), dtype=torch.float))
        self.ks = nn.Parameter(torch.zeros(size=(1, 1, nhid), dtype=torch.float))
        self.vs = nn.Parameter(torch.zeros(size=(1, 1, nhid), dtype=torch.float))
        self.qkvs = nn.Parameter(torch.zeros(size=(1, 3, nhid), dtype=torch.float))
        self.heads = heads
        self.nhid = nhid
        assert nhid % self.heads == 0, 'Heads must divide vector evenly'
        self.drop = nn.Dropout(dropout) if dropout else None
        self.gelu = GELU()
        self.q = nn.Linear(nhid, nhid) if q else None
        self.qln = LayerNorm(nhid, eps=1e-12)
        self.k = nn.Linear(nhid, nhid) if k else None
        self.v = nn.Linear(nhid, nhid) if v else None
        self.r = nn.Linear(2 * nhid, nhid) if r else None
        self.r_gate = nn.Parameter(torch.ones(size=(1, 1, nhid), dtype=torch.float))
        self.vq = None
        self.vq = Overparam(nhid)
        #from fastai.text.models import QRNNLayer
        #self.vq = QRNNLayer(input_size=nhid, hidden_size=nhid, save_prev_x=False, zoneout=0, window=1, output_gate=False, batch_first=False)
        self.vq_collapsed = False

    def vq_collapse(self):
        vs = torch.sigmoid(self.vs)
        #vs, _ = self.vq(vs)
        vs = self.vq(vs)
        self.vs.data = vs.data
        self.vq = None
        self.vq_collapsed = True

    def forward(self, query, key, value, attn_mask=None, batch_first=False, **kwargs):
        # tanh on the value allows us to flip the polarity of the output, helping use the full range
        # Discovered accidentally when I used QRNN_with_tanh_output(sigmoid(vs))
        #qs, ks, vs = torch.sigmoid(self.qs), torch.sigmoid(self.ks), self.vs
        qs, ks, vs = torch.sigmoid(self.qs), torch.sigmoid(self.ks), torch.sigmoid(self.vs)
        #qs, ks, vs = self.qs, self.ks, self.vs
        #vs = torch.tanh(self.vs)
        if self.vq:
            #vs, _ = self.vq(vs)
            vs = self.vq(vs)
            #qs, ks, vs = [x.reshape((1, 1, -1)) for x in self.vq(torch.sigmoid(self.qkvs))[0, :]]
        elif self.vq_collapsed:
            vs = self.vs
        #qs, ks, vs = self.qs, self.ks, self.vs
        #q = qs * query
        #if self.q: query = self.q(query)
        if self.q:
            query = self.q(query)
            query = self.qln(query.float())
        if self.k: key = self.k(key)
        if self.v: value = self.v(value)
        # This essentially scales everything to zero to begin with and then learns from there
        #q, k, v = self.qs * query, self.ks * key, self.vs * value
        q, k, v = qs * query, ks * key, vs * value
        #q, k, v = query, key, vs * value
        #q, k, v = qs * query, ks * key, value
        #k, v = ks * key, vs * value
        #q, k, v = query, key, value
        if self.drop:
            # We won't apply dropout to v as we can let the caller decide if dropout should be applied to the output
            # Applying dropout to q is equivalent to the same mask on k as they're "zipped"
            #q, k, v = self.drop(q), k, v
            q, k, v = self.drop(q), k, self.drop(v)

        original_q = q

        if not batch_first:
            q, k, v = q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)

        batch_size, query_len, nhid = q.size()
        assert nhid == self.nhid
        key_len = k.size(1)
        ###
        dim = self.nhid // self.heads
        q = q.view(batch_size, query_len, self.heads, dim).transpose(1, 2)
        k, v = [vec.view(batch_size, key_len, self.heads, dim).transpose(1, 2) for vec in [k, v]]

        mix, focus = attention(q, k, v, dropout=self.drop, attn_mask=attn_mask, **kwargs)
        mix = mix.transpose(1, 2).contiguous().view(batch_size, -1, self.nhid)
        if not batch_first:
            mix = mix.transpose(0, 1)

        if self.r:
            # The result should be transformed according to the query
            r = torch.cat([mix, original_q], dim=-1)
            if self.drop: r = self.drop(r)
            r = self.gelu(self.r(r))
            mix = torch.sigmoid(self.r_gate) * mix + r
            # BUG: This does _nothing_ as mix isn't set to r ...
            # But ... I got good results with this ... so ...
            # Let's leave it as is for right now ...
            # This does imply that I don't necessarily need complex post mixing ops

        return mix, focus

class PyTorchAttention(nn.Module):
    def __init__(self, nhid, q=True, k=False, v=False, heads=1, dropout=None):
        super().__init__()
        self.mha = nn.MultiheadAttention(nhid, heads, dropout=dropout)

    def forward(self, q, k, v, attn_mask=None):
        return self.mha(q, k, v, attn_mask=attn_mask)

class Block(nn.Module):
    def __init__(self, embed_dim, hidden_dim, heads=1, dropout=None, rnn=False, residual=True, use_attn=True):
        super().__init__()
        #self.attn = PyTorchAttention(embed_dim, heads=heads, dropout=dropout)
        self.attn = None
        if use_attn:
            self.attn = Attention(embed_dim, heads=heads, r=False, dropout=dropout)
        self.ff = Boom(embed_dim, hidden_dim, dropout=dropout, shortcut=True)
        self.lnstart = LayerNorm(embed_dim, eps=1e-12)
        self.lnmid = LayerNorm(embed_dim, eps=1e-12)
        self.lnmem = LayerNorm(embed_dim, eps=1e-12)
        self.lnout = LayerNorm(embed_dim, eps=1e-12)
        self.lnff = LayerNorm(embed_dim, eps=1e-12)
        self.lnxff = LayerNorm(embed_dim, eps=1e-12)
        self.drop = nn.Dropout(dropout)
        self.gelu = GELU()
        self.residual = residual

        self.rnn = None
        if rnn:
            self.rnn = nn.LSTM(input_size=embed_dim, hidden_size=embed_dim, batch_first=False)
            if rnn not in [True, False]:
                self.rnn = rnn

    def forward(self, h, pe, attn_mask, mem=None, hidden=None):
        new_mem = None

        h = self.lnstart(h)

        if self.rnn:
            x, new_hidden = self.rnn(h, None if hidden is None else hidden)
            #x = self.rnn_down(self.drop(x))

            # Trim the end off if the size is different
            ninp = h.shape[-1]
            z = torch.narrow(x, -1, 0, x.shape[-1] // ninp * ninp)
            # Divide the hidden size evenly into chunks
            z = x.view(*x.shape[:-1], x.shape[-1] // ninp, ninp)
            # Collapse the chunks through summation
            #h = h + self.drop(x).sum(dim=-2)
            x = self.drop(z).sum(dim=-2)
            #x = x + z.sum(dim=-2)

            h = h + x if self.residual else x.float()

        focus, new_mem = None, []
        if self.attn is not None:
            mh = self.lnmem(h)
            h = self.lnmid(h)

            if mem is not None:
                bigh = torch.cat([mem, mh], dim=0)
            else:
                bigh = mh
            new_mem = bigh[-len(pe):]

            q, k = h, bigh

            x, focus = checkpoint(self.attn, q, k, bigh, attn_mask)
            #x, focus = tcheckpoint(self.attn, q, k, bigh, attn_mask)
            x = self.drop(x)
            h = x + h

        if self.ff:
            h, x = self.lnff(h), self.lnxff(h)
            x = checkpoint(self.ff, x)
            #x = tcheckpoint(self.ff, h)
            x = self.drop(x)
            h = x + h

        return h, new_mem, new_hidden, focus

class SHARNN(nn.Module):
    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, dropouth=0.5, dropouti=0.5, dropoute=0.1, wdrop=0, tie_weights=False):
        super().__init__()
        embed_dim = ninp
        hidden_dim = nhid
        self.ninp, self.nhid = ninp, nhid
        self.nlayers = nlayers
        num_embeddings = ntoken
        self.num_max_positions = 5000 # 2500 # 5000 # 4096 # 2048 # 4096 + 1024 # 2048 # 5000 # 4096 # 1024 # 4096 # 512 # 1024 # 4096 # 4608 # 7168 # 8192 # 6144 # 4608 # 5000 # 4096 # 3072 # 8192 # 4096
        self.num_heads = 1 # 4
        num_layers = nlayers
        self.causal = True
        self.drop = nn.Dropout(dropout)
        self.idrop = nn.Dropout(dropouti)
        self.hdrop = nn.Dropout(dropouth)

        #from fastai.text.models import QRNN, QRNNLayer

        self.blocks = nn.ModuleList()
        for idx in range(num_layers):
            #rnn = True if idx in [0, num_layers - 1] else mid_rnn
            #rnn = rnns[0]
            #rnn = rnns[idx % 2]
            #rnn = rnns[idx]
            rnn = True
            self.blocks.append(Block(embed_dim, hidden_dim, self.num_heads, dropout=dropouth, rnn=rnn, residual=False, use_attn=True if idx == num_layers - 2 else False))

        #self.pos_emb = nn.Parameter(torch.zeros(size=(self.num_max_positions, 1, embed_dim), dtype=torch.float))
        self.pos_emb = [0] * self.num_max_positions
        #self.position_gates = torch.nn.ParameterList([nn.Parameter(torch.zeros(size=(1, 1, embed_dim), dtype=torch.float)) for _ in range(num_layers)])

        self.encoder = nn.Embedding(num_embeddings, embed_dim)
        self.decoder = nn.Linear(embed_dim, num_embeddings)
        if tie_weights:
            #if nhid != ninp:
            #    raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding, nn.LayerNorm)):
            module.weight.data.normal_(mean=0.0, std=0.1 / np.sqrt(self.ninp))

        if isinstance(module, (nn.Linear, nn.LayerNorm)) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, x, hidden=None, mems=None, padding_mask=None, return_h=True):
        """ Input has shape [seq length, batch] """
        e = self.encoder(x)
        e = self.idrop(e)

        if mems is not None:
            maxmem = self.num_max_positions - len(e)
            mems = [m[-maxmem:] for m in mems]

        total_length = len(x) + (len(mems[0]) if mems else 0)
        #pos_seq = torch.arange(self.num_max_positions - 1, -1, -1.0, device=e.device, dtype=torch.float)
        #pe = self.pos_emb(pos_seq)
        # #!&*!^$*&!*#&!YRUFEYDBW!^U#TEGWDBSTHTI!@UYEGDI^HJSTDGIQ
        pe = self.pos_emb #* 0
        #pe = self.dynamic_pe[:len(e)]
        #pe = self.idrop(pe)

        h = e

        new_hidden = []

        new_mems = []

        focus = []

        attn_mask = None
        if self.causal:
            attn_mask = torch.full((len(x), len(x)), -float('Inf'), device=h.device, dtype=h.dtype)
            attn_mask = torch.triu(attn_mask, diagonal=1)
            if mems:
                max_mems = max(len(m) for m in mems)
                happy = torch.zeros((len(x), max_mems), device=h.device, dtype=h.dtype)
                attn_mask = torch.cat([happy, attn_mask], dim=-1)

        for idx, block in enumerate(self.blocks):
            mem = mems[idx] if mems else None
            hid = hidden[idx] if hidden else None
            #p = torch.sigmoid(self.position_gates[idx]) * pe
            h, m, nh, f = block(h, pe, attn_mask=attn_mask, mem=mem, hidden=hid)
            #focus.append(f)
            new_hidden.append(nh)
            new_mems.append(m)

        h = self.drop(h)

        if return_h:
            return h, new_hidden, new_mems, None, None
        return h, new_hidden, new_mems

class GELU(nn.Module):
    """
    Paper Section 3.4, last paragraph notice that BERT used the GELU instead of RELU
    """
    def forward(self, x):
        #return torch.nn.functional.gelu(x.float())
        # The first approximation has more operations than the second
        # See https://arxiv.org/abs/1606.08415
        #return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        return x * torch.sigmoid(1.702 * x)

#@torch.jit.script
#def GELU(x):
#    return x * torch.sigmoid(1.702 * x)

class Boom(nn.Module):

    def __init__(self, d_model, dim_feedforward=2048, dropout=0.1, shortcut=False):
        super(Boom, self).__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout) if dropout else None
        if not shortcut:
            self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.shortcut = shortcut
        #self.act = nn.ReLU()
        self.act = GELU()
        #self.act = nn.Tanh()

    def forward(self, input):
        x = self.act(self.linear1(input))
        if self.dropout: x = self.dropout(x)
        if self.shortcut:
            # Trim the end off if the size is different
            ninp = input.shape[-1]
            x = torch.narrow(x, -1, 0, x.shape[-1] // ninp * ninp)
            # Divide the hidden size evenly into chunks
            x = x.view(*x.shape[:-1], x.shape[-1] // ninp, ninp)
            # Collapse the chunks through summation
            #h = h + self.drop(x).sum(dim=-2)
            z = x.sum(dim=-2)
        else:
            z = self.linear2(x)

        return z
