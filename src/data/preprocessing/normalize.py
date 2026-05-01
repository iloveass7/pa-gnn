"""
Per-tile min-max normalization.
Normalizes image intensity values to [0, 1] range with epsilon for numerical stability.
"""

import numpy as np
import torch


def per_tile_minmax(image, eps=1e-8):
    """
    Per-tile min-max normalization to [0, 1].
    
    Args:
        image: numpy array (H, W) or (H, W, C), any numeric dtype.
        eps: Small constant to avoid division by zero.
        
    Returns:
        Normalized numpy array, dtype float32, values in [0, 1].
    """
    image = image.astype(np.float32)
    imin = image.min()
    imax = image.max()
    return (image - imin) / (imax - imin + eps)


def per_tile_minmax_tensor(tensor, eps=1e-8):
    """
    Per-tile min-max normalization for PyTorch tensors.
    
    Args:
        tensor: torch.Tensor of shape (C, H, W) or (H, W).
        eps: Small constant to avoid division by zero.
        
    Returns:
        Normalized tensor, values in [0, 1].
    """
    tmin = tensor.min()
    tmax = tensor.max()
    return (tensor - tmin) / (tmax - tmin + eps)


def per_channel_minmax(image, eps=1e-8):
    """
    Per-channel min-max normalization (each channel normalized independently).
    
    Args:
        image: numpy array (H, W, C), any numeric dtype.
        eps: Small constant to avoid division by zero.
        
    Returns:
        Normalized numpy array, dtype float32, values in [0, 1].
    """
    image = image.astype(np.float32)
    result = np.empty_like(image)
    for c in range(image.shape[-1]):
        ch = image[..., c]
        cmin, cmax = ch.min(), ch.max()
        result[..., c] = (ch - cmin) / (cmax - cmin + eps)
    return result
