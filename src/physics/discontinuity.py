"""
Discontinuity feature extraction (D).
Computes absolute Laplacian of Gaussian (LoG) response and normalizes to [0, 1].
Implemented in PyTorch for fast batched processing.
"""

import math
import torch
import torch.nn.functional as F

def get_log_kernel(kernel_size=9, sigma=2.0, dtype=torch.float32, device='cpu'):
    """Generate a Laplacian of Gaussian (LoG) kernel."""
    pad = kernel_size // 2
    grid_x, grid_y = torch.meshgrid(
        torch.arange(-pad, pad + 1, dtype=dtype, device=device),
        torch.arange(-pad, pad + 1, dtype=dtype, device=device),
        indexing='ij'
    )
    
    var = sigma ** 2
    r2 = grid_x ** 2 + grid_y ** 2
    
    # LoG formula: -(1 / (pi * sigma^4)) * (1 - (x^2+y^2)/(2*sigma^2)) * exp(-(x^2+y^2)/(2*sigma^2))
    # Note: We take absolute response, so sign doesn't strictly matter, but this is the standard form
    norm = 1.0 / (math.pi * (var ** 2))
    log_kernel = -norm * (1.0 - r2 / (2.0 * var)) * torch.exp(-r2 / (2.0 * var))
    
    # Ensure zero mean to not respond to flat intensity areas
    log_kernel = log_kernel - log_kernel.mean()
    
    return log_kernel.view(1, 1, kernel_size, kernel_size)


def compute_discontinuity(image, kernel_size=9, sigma=2.0, eps=1e-8):
    """
    Compute discontinuity (D) via absolute LoG response.
    
    Args:
        image: torch.Tensor of shape (B, 1, H, W) or (1, H, W).
               Intensity values should be in [0, 1].
        kernel_size: Size of LoG kernel.
        sigma: Standard deviation of Gaussian.
        eps: Small constant for numerical stability.
               
    Returns:
        torch.Tensor of shape (B, 1, H, W) or (1, H, W) with discontinuity values in [0, 1].
    """
    is_3d = image.dim() == 3
    if is_3d:
        image = image.unsqueeze(0)
        
    device = image.device
    dtype = image.dtype
    
    log_kernel = get_log_kernel(kernel_size=kernel_size, sigma=sigma, dtype=dtype, device=device)
    
    pad = kernel_size // 2
    padded = F.pad(image, (pad, pad, pad, pad), mode='reflect')
    
    response = F.conv2d(padded, log_kernel)
    abs_response = torch.abs(response)
    
    # Per-tile min-max normalization
    B = abs_response.size(0)
    resp_flat = abs_response.view(B, -1)
    resp_min = resp_flat.min(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
    resp_max = resp_flat.max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
    
    discontinuity = (abs_response - resp_min) / (resp_max - resp_min + eps)
    
    if is_3d:
        discontinuity = discontinuity.squeeze(0)
        
    return discontinuity
