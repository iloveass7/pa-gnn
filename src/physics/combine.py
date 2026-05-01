"""
Combine physics features.
H_physics = w_s * S + w_r * R + w_d * D
"""

def combine_physics(slope, roughness, discontinuity, w_s=0.4, w_r=0.3, w_d=0.3):
    """
    Weighted combination of physics features to produce H_physics.
    Assumes all inputs are in [0, 1].
    
    Args:
        slope (torch.Tensor): S map
        roughness (torch.Tensor): R map
        discontinuity (torch.Tensor): D map
        w_s, w_r, w_d (float): Weights for S, R, D respectively. Should sum to 1.0.
        
    Returns:
        torch.Tensor: Combined H_physics map in [0, 1]
    """
    # Normalize weights to sum to 1.0 just in case
    total_w = w_s + w_r + w_d
    w_s, w_r, w_d = w_s / total_w, w_r / total_w, w_d / total_w
    
    h_physics = (w_s * slope) + (w_r * roughness) + (w_d * discontinuity)
    return h_physics
