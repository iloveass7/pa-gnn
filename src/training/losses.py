import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

class RiskLoss(nn.Module):
    def __init__(self, bce_weight=1.0, bce_hazard_weight=3.0, dice_weight=0.5, 
                 dice_threshold=0.7, tv_weight=0.1):
        super().__init__()
        self.bce_weight = bce_weight
        self.bce_hazard_weight = bce_hazard_weight
        self.dice_weight = dice_weight
        self.dice_threshold = dice_threshold
        self.tv_weight = tv_weight

    def forward(self, pred, target):
        if isinstance(pred, dict):
            pred = pred['h_final']
            
        # target is -1 for ignore regions
        valid_mask = (target >= 0).float()
        
        if valid_mask.sum() == 0:
            # All pixels are ignored — this should be rare. Log a warning.
            warnings.warn(
                f"RiskLoss: valid_mask is empty! target range=[{target.min():.2f}, {target.max():.2f}], "
                f"shape={target.shape}. Check label loading/remapping."
            )
            return {
                'loss': pred.sum() * 0.0, 
                'bce': torch.tensor(0.0, device=pred.device), 
                'dice': torch.tensor(0.0, device=pred.device), 
                'tv': torch.tensor(0.0, device=pred.device)
            }

        # 1. Weighted BCE Loss
        hazard_mask = (target > self.dice_threshold).float()
        weight_map = 1.0 + (self.bce_hazard_weight - 1.0) * hazard_mask
        weight_map = weight_map * valid_mask
        
        # Clamp predictions slightly for numerical stability
        pred_clamped = torch.clamp(pred, min=1e-6, max=1.0-1e-6)
        
        # Use valid_mask to zero out target where target < 0
        bce = F.binary_cross_entropy(pred_clamped, target * valid_mask, weight=weight_map, reduction='sum') / (valid_mask.sum() + 1e-8)
        
        # 2. Dice Loss on hazardous region
        t_hazard = hazard_mask * valid_mask
        p_hazard = pred * valid_mask
        
        intersection = (p_hazard * t_hazard).sum()
        union = p_hazard.sum() + t_hazard.sum()
        dice_loss = 1.0 - (2.0 * intersection + 1e-8) / (union + 1e-8)
        
        # 3. TV Smoothness Loss
        diff_h = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :]) * valid_mask[:, :, 1:, :]
        diff_w = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1]) * valid_mask[:, :, :, 1:]
        tv_loss = (diff_h.sum() + diff_w.sum()) / (valid_mask.sum() + 1e-8)
        
        total_loss = self.bce_weight * bce + self.dice_weight * dice_loss + self.tv_weight * tv_loss
        
        return {
            'loss': total_loss,
            'bce': bce.detach(),
            'dice': dice_loss.detach(),
            'tv': tv_loss.detach()
        }
