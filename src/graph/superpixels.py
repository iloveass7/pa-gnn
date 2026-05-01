import numpy as np
from skimage.segmentation import slic
import torch

def compute_superpixels(image, n_segments=300, compactness=10.0, sigma=1.0):
    """
    Compute SLIC superpixels on the original image.
    Args:
        image: torch.Tensor (1, H, W) or (3, H, W), values in [0, 1]
    Returns:
        label_map: np.ndarray (H, W) of superpixel labels
    """
    # Convert to numpy and channels-last
    img_np = image.permute(1, 2, 0).cpu().numpy()
    
    # If 1 channel, convert back to 2D
    if img_np.shape[2] == 1:
        img_np = img_np[:, :, 0]
        
    label_map = slic(img_np, n_segments=n_segments, compactness=compactness, sigma=sigma, start_label=0)
    return label_map
