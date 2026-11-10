import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import ChebConv, global_mean_pool

class SpectralGCN(nn.Module):
    """Two Chebyshev graph-convolution layers and graph-level classifier."""
    def __init__(self, in_channels, hidden=64, k=3, dropout=0.5):
        super().__init__()
        self.conv1 = ChebConv(in_channels, hidden, K=k)
        self.conv2 = ChebConv(hidden, hidden, K=k)
        self.fc = nn.Linear(hidden, 2)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight, batch):
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index, edge_weight))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_mean_pool(x, batch)
        return F.log_softmax(self.fc(x), dim=1)
