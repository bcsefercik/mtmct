"""
Graph Attention Networks in DGL using SPMV optimization.
References
----------
Paper: https://arxiv.org/abs/1710.10903
Author's code: https://github.com/PetarV-/GAT
Pytorch implementation: https://github.com/Diego999/pyGAT
"""

import torch
import torch.nn as nn
import dgl.function as fn
from dgl.nn.pytorch import edge_softmax, GATConv

import torch.nn.functional as F


class BCS(nn.Module):
    def __init__(self,
        g=None,
        n_nodes = 4,  
        num_layers=3,
        in_dim=128,
        num_hidden=89,
        g_out_dim=128,
        num_heads=8,
        num_out_heads=1,
        activation=F.elu,
        feat_drop=.6,
        attn_drop=.6,
        negative_slope=.2,
        residual=False):
        super(BCS, self).__init__()
        self.g = g
        heads = ([num_heads] * num_layers) + [num_out_heads]
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        # input projection (no residual)
        self.gat_layers.append(GATConv(
            in_dim, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GATConv(
                num_hidden * heads[l-1], num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation))
        # output projection
        self.gat_layers.append(GATConv(
            num_hidden * heads[-2], g_out_dim, heads[-1],
            feat_drop, attn_drop, negative_slope, residual, None))

        self.fc1 = nn.Sequential(
            nn.Linear(n_nodes*g_out_dim + 128, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 1)
        )


    def forward(self, h, f):
        for l in range(self.num_layers):
            h = self.gat_layers[l](self.g, h).flatten(1)
        h = self.gat_layers[-1](self.g, h).mean(1)
        h = h.view(-1)
        r = self.fc1(torch.cat((h, f), 0))
        # # output projection
        # logits = 
        return r