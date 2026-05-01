"""
Slope feature extraction (S).
Computes Sobel gradient magnitude and normalizes to [0, 1].
Implemented in PyTorch for fast batched processing on CPU/GPU.
"""

import torch
import torch.nn.functional as F

from src.data.preprocessing.normalize import per_tile_minmax_tensor

def compute_slope(image, eps=1e-8):
    """
    Compute slope (S) via Sobel gradient magnitude.
    
    Args:
        image: torch.Tensor of shape (B, 1, H, W) or (1, H, W).
               Intensity values should be in [0, 1].
        eps: Small constant for numerical stability during normalization.
               
    Returns:
        torch.Tensor of shape (B, 1, H, W) or (1, H, W) with slope values in [0, 1].
    """
    is_3d = image.dim() == 3
    if is_3d:
        image = image.unsqueeze(0)  # (1, 1, H, W)
        
    device = image.device
    dtype = image.dtype
    
    # Sobel kernels
    sobel_x = torch.tensor([[-1., 0., 1.],
                            [-2., 0., 2.],
                            [-1., 0., 1.]], dtype=dtype, device=device).view(1, 1, 3, 3)
                            
    sobel_y = torch.tensor([[-1., -2., -1.],
                            [ 0.,  0.,  0.],
                            [ 1.,  2.,  1.]], dtype=dtype, device=device).view(1, 1, 3, 3)
                            
    # Compute gradients (use reflection padding to avoid edge artifacts)
    padded = F.pad(image, (1, 1, 1, 1), mode='reflect')
    gx = F.conv2d(padded, sobel_x)
    gy = F.conv2d(padded, sobel_y)
    
    # Gradient magnitude
    g_mag = torch.sqrt(gx**2 + gy**2 + eps)
    
    # Per-tile min-max normalization
    B = g_mag.size(0)
    g_mag_flat = g_mag.view(B, -1)
    g_min = g_mag_flat.min(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
    g_max = g_mag_flat.max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
    
    slope = (g_mag - g_min) / (g_max - g_min + eps)
    
    if is_3d:
        slope = slope.squeeze(0)
        
    return slope
