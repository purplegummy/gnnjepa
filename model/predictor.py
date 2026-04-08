import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv


class Predictor(nn.Module):
    def __init__(self, in_dim=33, hidden_dim=64, out_dim=32):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)

    def forward(self, z, action, edge_index):
        # concatenate node features with action before applying GCN
        x = torch.cat([z, action], dim=-1)  # [num_nodes, in_dim]
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x  # [num_nodes, out_dim]

