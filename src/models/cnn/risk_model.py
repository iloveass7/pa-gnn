import torch.nn as nn
from src.models.cnn.mobilenetv3 import MobileNetV3Encoder
from src.models.cnn.deeplabv3plus import DeepLabV3PlusDecoder
from src.models.cnn.risk_head import RiskHead

class RiskModel(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        if cfg is None:
            pretrained = True
            freeze_bn = False
            aspp_rates = [6, 12, 18]
            aspp_out_channels = 256
            low_level_channels = 24
            low_level_out_channels = 48
        else:
            pretrained = cfg.model.encoder.pretrained
            freeze_bn = cfg.model.encoder.freeze_bn
            aspp_rates = cfg.model.decoder.aspp_rates
            aspp_out_channels = cfg.model.decoder.aspp_out_channels
            low_level_channels = cfg.model.decoder.low_level_channels
            low_level_out_channels = cfg.model.decoder.low_level_out_channels

        self.encoder = MobileNetV3Encoder(pretrained=pretrained, freeze_bn=freeze_bn)
        
        # MobileNetV3 large defaults: low_level=24 (from layer 3), high_level=960 (from layer 16)
        self.decoder = DeepLabV3PlusDecoder(
            in_channels=960,
            low_level_channels=low_level_channels,
            aspp_out_channels=aspp_out_channels,
            low_level_out_channels=low_level_out_channels,
            atrous_rates=aspp_rates
        )
        
        self.head = RiskHead(in_channels=256, out_channels=1)

    def forward(self, x):
        target_size = x.shape[2:]
        low_level, high_level = self.encoder(x)
        dec_out = self.decoder(low_level, high_level)
        out = self.head(dec_out, target_size=target_size)
        return out
