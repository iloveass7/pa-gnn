import torch
import torch.nn as nn
import torch.nn.functional as F

class RiskHead(nn.Module):
    def __init__(self, in_channels=256, out_channels=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1)
        
    def forward(self, x, target_size=None):
        x = self.conv(x)
        if target_size is not None:
            x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        return torch.sigmoid(x)
