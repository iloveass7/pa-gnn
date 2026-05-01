import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights

class MobileNetV3Encoder(nn.Module):
    def __init__(self, pretrained=True, freeze_bn=False):
        super().__init__()
        weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        model = mobilenet_v3_large(weights=weights)
        
        self.features = model.features
        
        # We need to extract low-level and high-level features.
        # MobileNetV3 Large features structure:
        # Layer 3: output is C=24, stride=4 (low-level features)
        # Layer 16: output is C=960, stride=32 (high-level features)
        
        self.low_level_idx = 3
        self.high_level_idx = 16
        
        if freeze_bn:
            self._freeze_bn()

    def _freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                for p in m.parameters():
                    p.requires_grad = False

    def forward(self, x):
        low_level_feat = None
        for i, module in enumerate(self.features):
            x = module(x)
            if i == self.low_level_idx:
                low_level_feat = x
        return low_level_feat, x
