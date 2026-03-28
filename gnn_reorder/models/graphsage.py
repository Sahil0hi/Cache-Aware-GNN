"""
models/graphsage.py
Two-layer GraphSAGE model for node classification.

Architecture:
  SAGEConv(in_channels, hidden) -> ReLU -> Dropout
  SAGEConv(hidden, num_classes)
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class GraphSAGE(torch.nn.Module):
    """Two-layer GraphSAGE for node classification."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x  # raw logits; use F.log_softmax externally if needed
