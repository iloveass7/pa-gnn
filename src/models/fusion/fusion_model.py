import torch
import torch.nn as nn
from src.models.cnn.risk_model import RiskModel
from src.data.transforms.physics_features import PhysicsFeatureExtractor
from src.models.fusion.adaptive_fusion import AdaptiveFusion

class EndToEndFusionModel(nn.Module):
    def __init__(self, cnn_cfg, phys_cfg, fusion_cfg=None, freeze_cnn=True):
        super().__init__()
        self.cnn = RiskModel(cnn_cfg)
        self.physics_extractor = PhysicsFeatureExtractor.from_config(phys_cfg)
        self.fusion = AdaptiveFusion()
        
        self.freeze_cnn = freeze_cnn
        if self.freeze_cnn:
            for p in self.cnn.parameters():
                p.requires_grad = False
            self.cnn.eval()
            
    def train(self, mode=True):
        super().train(mode)
        if self.freeze_cnn:
            self.cnn.eval()
            
    def forward(self, image):
        # 1. Physics features (always frozen/no grad)
        with torch.no_grad():
            phys_dict = self.physics_extractor(image)
            h_physics = phys_dict['H_physics']
            
        # 2. Learned features
        if self.freeze_cnn:
            with torch.no_grad():
                h_learned = self.cnn(image)
        else:
            h_learned = self.cnn(image)
            
        # 3. Fusion
        fusion_out = self.fusion(h_physics, h_learned, image)
        
        return {
            'S': phys_dict['S'],
            'R': phys_dict['R'],
            'D': phys_dict['D'],
            'h_physics': h_physics,
            'h_learned': h_learned,
            'alpha': fusion_out['alpha'],
            'h_final': fusion_out['h_final']
        }
