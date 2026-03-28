"""
models/gat.py
Two-layer Graph Attention Network (GAT) for node classification.

Architecture:
  GATConv(in_channels, hidden // heads, heads=heads) -> ELU -> Dropout
  GATConv(hidden, num_classes, heads=1, concat=False)
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GAT(torch.nn.Module):
    """Two-layer GAT for node classification."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        heads: int = 8,
        dropout: float = 0.6,
    ):
        super().__init__()
        # First layer: multi-head attention, heads are concatenated
        self.conv1 = GATConv(
            in_channels,
            hidden_channels // heads,
            heads=heads,
            dropout=dropout,
            concat=True,
        )
        # Second layer: single-head, output is averaged
        self.conv2 = GATConv(
            hidden_channels,
            out_channels,
            heads=1,
            dropout=dropout,
            concat=False,
        )
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x  # raw logits
