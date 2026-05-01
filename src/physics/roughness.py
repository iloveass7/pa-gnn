"""
Roughness feature extraction (R).
Computes local standard deviation in a sliding window and normalizes to [0, 1].
Implemented in PyTorch for fast batched processing.
"""

import torch
import torch.nn.functional as F

def compute_roughness(image, window_size=7, eps=1e-8):
    """
    Compute roughness (R) via local standard deviation.
    
    Args:
        image: torch.Tensor of shape (B, 1, H, W) or (1, H, W).
               Intensity values should be in [0, 1].
        window_size: Size of the sliding window (default 7).
        eps: Small constant for numerical stability.
               
    Returns:
        torch.Tensor of shape (B, 1, H, W) or (1, H, W) with roughness values in [0, 1].
    """
    is_3d = image.dim() == 3
    if is_3d:
        image = image.unsqueeze(0)  # (1, 1, H, W)
        
    # E[X^2] - (E[X])^2
    kernel = torch.ones((1, 1, window_size, window_size), dtype=image.dtype, device=image.device)
    kernel = kernel / (window_size * window_size)
    
    pad = window_size // 2
    padded = F.pad(image, (pad, pad, pad, pad), mode='reflect')
    
    mean_x = F.conv2d(padded, kernel)
    mean_x2 = F.conv2d(padded**2, kernel)
    
    # Local variance (clamp to 0 to avoid negative values due to float precision)
    var = torch.clamp(mean_x2 - mean_x**2, min=0.0)
    std = torch.sqrt(var + eps)
    
    # Per-tile min-max normalization
    B = std.size(0)
    std_flat = std.view(B, -1)
    std_min = std_flat.min(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
    std_max = std_flat.max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
    
    roughness = (std - std_min) / (std_max - std_min + eps)
    
    if is_3d:
        roughness = roughness.squeeze(0)
        
    return roughness
