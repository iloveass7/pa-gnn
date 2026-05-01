import torch
import torch.nn as nn
import torch.nn.functional as F

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates):
        super().__init__()
        modules = []
        # 1x1 conv
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ))
        
        # Atrous convolutions
        for rate in atrous_rates:
            modules.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=rate, dilation=rate, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))
            
        # Global Average Pooling
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.convs = nn.ModuleList(modules)
        
        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels + out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        
        global_feat = self.gap(x)
        global_feat = F.interpolate(global_feat, size=x.shape[2:], mode='bilinear', align_corners=False)
        res.append(global_feat)
        
        res = torch.cat(res, dim=1)
        return self.project(res)

class DeepLabV3PlusDecoder(nn.Module):
    def __init__(self, in_channels=960, low_level_channels=24, aspp_out_channels=256, 
                 low_level_out_channels=48, atrous_rates=(6, 12, 18)):
        super().__init__()
        
        self.aspp = ASPP(in_channels, aspp_out_channels, atrous_rates)
        
        self.low_level_conv = nn.Sequential(
            nn.Conv2d(low_level_channels, low_level_out_channels, 1, bias=False),
            nn.BatchNorm2d(low_level_out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.fusion = nn.Sequential(
            nn.Conv2d(aspp_out_channels + low_level_out_channels, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

    def forward(self, low_level_feat, high_level_feat):
        high_level_feat = self.aspp(high_level_feat)
        low_level_feat = self.low_level_conv(low_level_feat)
        
        high_level_feat = F.interpolate(high_level_feat, size=low_level_feat.shape[2:], mode='bilinear', align_corners=False)
        
        out = torch.cat([high_level_feat, low_level_feat], dim=1)
        out = self.fusion(out)
        return out
