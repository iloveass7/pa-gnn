import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv

class PAGATv2(torch.nn.Module):
    """
    Physics-Aware Graph Attention Network v2 (PA-GNN).
    """
    def __init__(self, cfg):
        super().__init__()
        
        self.cfg = cfg
        m_cfg = cfg.model
        
        # Layer 1
        self.conv1 = GATv2Conv(
            in_channels=m_cfg.layer1.in_channels,
            out_channels=m_cfg.layer1.out_channels,
            heads=m_cfg.layer1.heads,
            concat=m_cfg.layer1.concat,
            dropout=m_cfg.layer1.dropout,
            edge_dim=1  # edge weights
        )
        
        self.act1 = nn.ELU() if m_cfg.layer1.activation == "elu" else nn.ReLU()
        self.drop1 = nn.Dropout(m_cfg.layer1.dropout)
        
        # Layer 2
        self.conv2 = GATv2Conv(
            in_channels=m_cfg.layer2.in_channels,
            out_channels=m_cfg.layer2.out_channels,
            heads=m_cfg.layer2.heads,
            concat=m_cfg.layer2.concat,
            dropout=m_cfg.layer2.dropout,
            edge_dim=1  # edge weights
        )
        
        self.act2 = nn.ELU() if m_cfg.layer2.activation == "elu" else nn.ReLU()
        self.drop2 = nn.Dropout(m_cfg.layer2.dropout)
        
        # Output Head
        self.out_linear = nn.Linear(m_cfg.output.in_features, m_cfg.output.out_features)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, edge_index, edge_attr):
        """
        Forward pass for PyG Data.
        """
        if edge_attr is not None and edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(-1)
            
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = self.act1(x)
        x = self.drop1(x)
        
        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        x = self.act2(x)
        x = self.drop2(x)
        
        x = self.out_linear(x)
        return self.sigmoid(x).squeeze(-1) # return shape (N,)
