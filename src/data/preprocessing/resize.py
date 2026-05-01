"""
Image and label resizing utilities.
Images use bilinear interpolation; labels use nearest-neighbour to preserve integer classes.
"""

import numpy as np
from PIL import Image


def resize_image(image, target_size, interpolation="bilinear"):
    """
    Resize an image using bilinear interpolation (default).
    
    Args:
        image: numpy array (H, W) or (H, W, C), dtype float32 or uint8.
        target_size: int (square) or tuple (H, W).
        interpolation: "bilinear" or "nearest".
        
    Returns:
        Resized numpy array with same number of channels.
    """
    if isinstance(target_size, int):
        target_size = (target_size, target_size)
    
    h, w = target_size
    
    interp_map = {
        "bilinear": Image.BILINEAR,
        "nearest": Image.NEAREST,
        "bicubic": Image.BICUBIC,
        "lanczos": Image.LANCZOS,
    }
    pil_interp = interp_map.get(interpolation, Image.BILINEAR)
    
    is_float = image.dtype in [np.float32, np.float64]
    
    if image.ndim == 2:
        pil_img = Image.fromarray(
            image if not is_float else (image * 255).clip(0, 255).astype(np.uint8),
            mode='L'
        )
        resized = pil_img.resize((w, h), pil_interp)
        result = np.array(resized, dtype=np.float32)
        if is_float:
            result = result / 255.0
        return result
    else:
        # Multi-channel
        if is_float:
            arr = (image * 255).clip(0, 255).astype(np.uint8)
        else:
            arr = image
        pil_img = Image.fromarray(arr)
        resized = pil_img.resize((w, h), pil_interp)
        result = np.array(resized, dtype=np.float32)
        if is_float:
            result = result / 255.0
        return result


def resize_label(label, target_size):
    """
    Resize a label mask using nearest-neighbour interpolation.
    This preserves integer class values without interpolation artifacts.
    
    Args:
        label: numpy array (H, W), dtype uint8 or int.
        target_size: int (square) or tuple (H, W).
        
    Returns:
        Resized label array, same dtype as input.
    """
    if isinstance(target_size, int):
        target_size = (target_size, target_size)
    
    h, w = target_size
    original_dtype = label.dtype
    
    pil_img = Image.fromarray(label.astype(np.uint8), mode='L')
    resized = pil_img.resize((w, h), Image.NEAREST)
    return np.array(resized, dtype=original_dtype)


def resize_risk_map(risk_map, target_size):
    """
    Resize a continuous risk map using bilinear interpolation.
    Handles the ignore value (-1) by interpolating only valid regions.
    
    Args:
        risk_map: numpy array (H, W), dtype float32. -1 = ignore.
        target_size: int (square) or tuple (H, W).
        
    Returns:
        Resized risk map, float32. Ignore regions preserved as -1.
    """
    if isinstance(target_size, int):
        target_size = (target_size, target_size)
    
    h, w = target_size
    ignore_mask = risk_map < 0
    
    # Replace ignore with 0 for interpolation, then restore
    safe_map = risk_map.copy()
    safe_map[ignore_mask] = 0.0
    
    # Resize the risk values (bilinear)
    pil_risk = Image.fromarray((safe_map * 255).clip(0, 255).astype(np.uint8), mode='L')
    resized_risk = np.array(pil_risk.resize((w, h), Image.BILINEAR), dtype=np.float32) / 255.0
    
    # Resize the ignore mask (nearest) to preserve boundaries
    pil_mask = Image.fromarray(ignore_mask.astype(np.uint8) * 255, mode='L')
    resized_mask = np.array(pil_mask.resize((w, h), Image.NEAREST)) > 127
    
    resized_risk[resized_mask] = -1.0
    return resized_risk
