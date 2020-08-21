import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.W = nn.Linear(in_features, out_features, bias=False)
        # self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.weight, gain=1.414)
        # self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        self.a1 = nn.Parameter(torch.zeros(size=(out_features, 1)))
        self.a2 = nn.Parameter(torch.zeros(size=(out_features, 1)))
        nn.init.xavier_uniform_(self.a1.data, gain=1.414)
        nn.init.xavier_uniform_(self.a2.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        h = self.W(input)
        # [batch_size, N, out_features]
        batch_size, N,  _ = h.size()
        middle_result1 = torch.matmul(h, self.a1).expand(-1, -1, N)
        middle_result2 = torch.matmul(h, self.a2).expand(-1, -1, N).transpose(1, 2)
        e = self.leakyrelu(middle_result1 + middle_result2)
        attention = e.masked_fill(adj == 0, -1e9)
        attention = F.softmax(attention, dim=2)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads, layer):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.layer = layer
        if self.layer == 1:
            self.attentions = [GraphAttentionLayer(nfeat, nclass, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        else:
            self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
            self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)   
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
            
    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        if self.layer == 1:
            x = torch.stack([att(x, adj) for att in self.attentions], dim=2)
            x = x.sum(2)
            x = F.dropout(x, self.dropout, training=self.training)
            return F.log_softmax(x, dim=2)
        else:
            x = torch.cat([att(x, adj) for att in self.attentions], dim=2)
            x = F.dropout(x, self.dropout, training=self.training)
            x = F.elu(self.out_att(x, adj))
            return F.log_softmax(x, dim=2)




