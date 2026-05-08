import numpy as np
from skimage.segmentation import slic
import torch

def compute_superpixels(image, n_segments=300, compactness=10.0, sigma=1.0):
    """
    Compute SLIC superpixels on the original image.
    Args:
        image: torch.Tensor (C, H, W) or (1, C, H, W), values in [0, 1]
    Returns:
        label_map: np.ndarray (H, W) of superpixel labels (1-indexed, background=0 unused)
    """
    # Handle batched input (1, C, H, W) → (C, H, W)
    if image.dim() == 4:
        image = image[0]

    # Convert (C, H, W) → (H, W, C) for skimage
    img_np = image.permute(1, 2, 0).cpu().numpy()
    
    # If 1 channel, convert to 2D (H, W)
    if img_np.shape[2] == 1:
        img_np = img_np[:, :, 0]
        
    # start_label=1 so regionprops treats 0 as background and doesn't skip any
    # superpixel region. With start_label=0, the label-0 superpixel would be
    # silently excluded by regionprops (it expects 1-indexed foreground labels).
    label_map = slic(img_np, n_segments=n_segments, compactness=compactness,
                     sigma=sigma, start_label=1)
    return label_map
