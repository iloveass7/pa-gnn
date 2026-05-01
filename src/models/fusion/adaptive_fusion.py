import torch
import torch.nn as nn

class AdaptiveFusion(nn.Module):
    """
    Adaptive Hybrid Risk Fusion Model.
    Dynamically merges H_physics and H_learned into H_final using a learned spatial attention map (alpha).
    """
    def __init__(self, in_channels=3):
        super().__init__()
        
        # 3-layer lightweight CNN
        self.net = nn.Sequential(
            # Conv(3 -> 16, 3x3) -> ReLU
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # Conv(16 -> 8, 3x3) -> ReLU
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # Conv(8 -> 1, 1x1) -> Sigmoid
            nn.Conv2d(8, 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, h_physics, h_learned, image):
        """
        Forward pass to compute alpha map and final fused risk map.
        
        Args:
            h_physics: Tensor of shape (B, 1, H, W) - Physics risk map
            h_learned: Tensor of shape (B, 1, H, W) - Learned CNN risk map
            image: Tensor of shape (B, C, H, W) or (B, 1, H, W) - Original image
            
        Returns:
            dict containing 'alpha' and 'h_final'
        """
        if image.size(1) == 3:
            image_gray = image.mean(dim=1, keepdim=True)
        else:
            image_gray = image
            
        # Stack inputs: [H_physics | H_learned | original_image]
        x = torch.cat([h_physics, h_learned, image_gray], dim=1)
        
        # Compute spatial trust map for learned signal (alpha)
        alpha = self.net(x)
        
        # H_final = alpha * H_learned + (1 - alpha) * H_physics
        h_final = alpha * h_learned + (1.0 - alpha) * h_physics
        
        return {
            'alpha': alpha,
            'h_final': h_final
        }

def get_static_fusion(h_physics, h_learned, alpha=0.5):
    """Static baseline fusion (B4)."""
    return alpha * h_learned + (1.0 - alpha) * h_physics
