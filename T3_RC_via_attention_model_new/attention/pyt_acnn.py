from torch import nn
import torch.nn.functional as F
from attention.utils import *


class ACNN(nn.Module):
    def __init__(self, opt, embedding):
        super(ACNN, self).__init__()
        self.opt = opt
        self.dw = embedding.shape[1]
        self.vac_len = embedding.shape[0]
        self.d = self.dw + 2 * self.opt.DP
        self.p = (self.opt.K - 1) // 2
        self.x_embedding = nn.Embedding(self.vac_len, self.dw)
        self.x_embedding.weight = nn.Parameter(torch.from_numpy(embedding))
        # self.e1_embedding = nn.Embedding(self.vac_len, self.dw)
        # self.e1_embedding.weight = nn.Parameter(torch.from_numpy(embedding))
        # self.e2_embedding = nn.Embedding(self.vac_len, self.dw)
        # self.e2_embedding.weight = nn.Parameter(torch.from_numpy(embedding))
        self.dist_embedding = nn.Embedding(self.opt.NP, self.opt.DP)
        self.rel_weight = nn.Parameter(torch.randn(self.opt.NR, self.opt.DC))
        self.dropout = nn.Dropout(self.opt.KP)
        self.conv = nn.Conv2d(1, self.opt.DC, (self.opt.K, self.d), (1, self.d), (self.p, 0), bias=True)
        self.U = nn.Parameter(torch.randn(self.opt.DC, self.opt.NR))
        self.max_pool = nn.MaxPool1d(self.opt.N, stride=1)

    def input_attention(self, input_tuple, is_training=True):
        x, e1, e1d2, e2, e2d1, zd, d1, d2 = input_tuple
        x_emb = self.x_embedding(x)  # (bs, n, dw)
        e1_emb = self.x_embedding(e1)
        e2_emb = self.x_embedding(e2)
        # zd_emb = self.dist_embedding(zd)
        # e1d2_emb = self.dist_embedding(e1d2)
        # e2d1_emb = self.dist_embedding(e2d1)
        dist1_emb = self.dist_embedding(d1)
        dist2_emb = self.dist_embedding(d2)
        x_cat = torch.cat((x_emb, dist1_emb, dist2_emb), 2)
        # e1_cat = torch.cat((e1_emb, zd_emb, e1d2_emb), 2)
        # e2_cat = torch.cat((e2_emb, e2d1_emb, zd_emb), 2)
        # if is_training:
        #     x_cat = self.dropout(x_cat)
        ine1_aw = F.softmax(torch.bmm(x_emb, e1_emb.transpose(2, 1)), 1)  # (bs, n, 1)
        ine2_aw = F.softmax(torch.bmm(x_emb, e2_emb.transpose(2, 1)), 1)
        # ine1_aw = F.softmax(torch.bmm(x_cat, e1_cat.transpose(2, 1)), 1)  # (bs, n, 1)
        # ine2_aw = F.softmax(torch.bmm(x_cat, e2_cat.transpose(2, 1)), 1)
        in_aw = (ine1_aw + ine2_aw) / 2
        R = torch.mul(x_cat, in_aw)
        return R

    # def attentive_pooling(self, R_star, all_y):
    #     rel_emb = torch.mm(all_y, self.rel_weight)  # (NR, NR) * (NR, DC)
    #     RU = torch.matmul(R_star.transpose(2, 1), self.U)  # (bs, n, nr)
    #     G = torch.matmul(RU, rel_emb)  # (bs, n, dc)
    #     AP = F.softmax(G, dim=1)
    #     RA = torch.mul(R_star, AP.transpose(2, 1))
    #     wo = self.max_pool(RA).squeeze(-1)
    #     return wo, self.rel_weight

    def attentive_pooling(self, R_star):
        RU = torch.matmul(R_star.transpose(2, 1), self.U)  # (bs, n, nr)
        G = torch.matmul(RU, self.rel_weight)  # (bs, n, dc)
        AP = F.softmax(G, dim=1)
        RA = torch.mul(R_star, AP.transpose(2, 1))
        wo = self.max_pool(RA).squeeze(-1)
        return wo

    def forward(self, input_tuple, is_training=True):
        R = self.input_attention(input_tuple, is_training)
        R_star = self.conv(R.unsqueeze(1)).squeeze(-1)  # (bs, dc, n)
        R_star = torch.tanh(R_star)
        wo = self.attentive_pooling(R_star)
        return wo, self.rel_weight


class DistanceLoss(nn.Module):
    def __init__(self, nr, margin=1):
        super(DistanceLoss, self).__init__()
        self.nr = nr
        self.margin = margin

    def forward(self, wo, rel_weight, in_y, all_y):
        wo_norm = F.normalize(wo)  # (bs, dc)  in_y (bs, nr)
        wo_norm_tile = wo_norm.unsqueeze(1).repeat(1, in_y.size()[-1], 1)  # (bs, nr, dc)
        rel_emb = torch.mm(in_y, rel_weight)  # (bs, dc)
        ay_emb = torch.mm(all_y, rel_weight)  # (nr, dc)
        gt_dist = torch.norm(wo_norm - rel_emb, 2, 1)  # (bs, 1)
        all_dist = torch.norm(wo_norm_tile - ay_emb, 2, 2)  # (bs, nr)
        masking_y = torch.mul(in_y, 10000)
        _t_dist = torch.min(torch.add(all_dist, masking_y), 1)[0]
        loss = torch.mean(self.margin + gt_dist - _t_dist)
        return loss
