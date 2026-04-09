import torch.nn as nn
from torch_geometric.nn import GCNConv


class Encoder(nn.Module):
    def __init__(self, in_dim=3, hidden_dim=64, out_dim=32):
        super().__init__()
        # each node has 7 features: [S, I, R] one-hot + [population, hosp_capacity, density, elderly_frac]
        # sum neighbor features, add self features, then linear transformation
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x  # [num_nodes, out_dim]

