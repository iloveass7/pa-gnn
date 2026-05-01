"""
Wrapper to compute all physics features (S, R, D, and H_physics) from an image tensor.
"""

import torch
import torch.nn as nn

from src.physics.slope import compute_slope
from src.physics.roughness import compute_roughness
from src.physics.discontinuity import compute_discontinuity
from src.physics.combine import combine_physics

class PhysicsFeatureExtractor(nn.Module):
    """
    Wrapper module to compute S, R, D, and H_physics from an image tensor.
    Can be applied on-the-fly during data loading or in the forward pass.
    """
    def __init__(
        self,
        eps=1e-8,
        slope_enabled=True,
        roughness_enabled=True,
        roughness_window=7,
        discontinuity_enabled=True,
        discontinuity_kernel=9,
        discontinuity_sigma=2.0,
        w_s=0.4,
        w_r=0.3,
        w_d=0.3
    ):
        super().__init__()
        self.eps = eps
        
        self.slope_enabled = slope_enabled
        
        self.roughness_enabled = roughness_enabled
        self.roughness_window = roughness_window
        
        self.discontinuity_enabled = discontinuity_enabled
        self.discontinuity_kernel = discontinuity_kernel
        self.discontinuity_sigma = discontinuity_sigma
        
        self.w_s = w_s
        self.w_r = w_r
        self.w_d = w_d

    @classmethod
    def from_config(cls, cfg):
        """Create from physics.yaml config."""
        phys_cfg = cfg.physics
        return cls(
            eps=phys_cfg.eps,
            slope_enabled=phys_cfg.slope.enabled,
            roughness_enabled=phys_cfg.roughness.enabled,
            roughness_window=phys_cfg.roughness.window_size,
            discontinuity_enabled=phys_cfg.discontinuity.enabled,
            discontinuity_kernel=phys_cfg.discontinuity.kernel_size,
            discontinuity_sigma=phys_cfg.discontinuity.sigma,
            w_s=phys_cfg.weights.w_s,
            w_r=phys_cfg.weights.w_r,
            w_d=phys_cfg.weights.w_d
        )

    @torch.no_grad()
    def forward(self, image):
        """
        Compute physics features.
        
        Args:
            image: torch.Tensor of shape (B, C, H, W) or (C, H, W).
                   Intensity values in [0, 1].
                   
        Returns:
            dict containing:
                'S': Slope tensor (B, 1, H, W) or (1, H, W)
                'R': Roughness tensor (B, 1, H, W) or (1, H, W)
                'D': Discontinuity tensor (B, 1, H, W) or (1, H, W)
                'H_physics': Combined tensor (B, 1, H, W) or (1, H, W)
        """
        is_3d = image.dim() == 3
        if is_3d:
            image = image.unsqueeze(0)
            
        # Convert RGB to grayscale by averaging if necessary
        if image.size(1) == 3:
            gray_image = image.mean(dim=1, keepdim=True)
        else:
            gray_image = image

        B, _, H, W = gray_image.shape
        device = gray_image.device
        dtype = gray_image.dtype
        
        # S
        if self.slope_enabled:
            S = compute_slope(gray_image, eps=self.eps)
        else:
            S = torch.zeros((B, 1, H, W), dtype=dtype, device=device)
            
        # R
        if self.roughness_enabled:
            R = compute_roughness(gray_image, window_size=self.roughness_window, eps=self.eps)
        else:
            R = torch.zeros((B, 1, H, W), dtype=dtype, device=device)
            
        # D
        if self.discontinuity_enabled:
            D = compute_discontinuity(gray_image, kernel_size=self.discontinuity_kernel, sigma=self.discontinuity_sigma, eps=self.eps)
        else:
            D = torch.zeros((B, 1, H, W), dtype=dtype, device=device)
            
        # H_physics
        H_phys = combine_physics(S, R, D, self.w_s, self.w_r, self.w_d)
        
        if is_3d:
            S = S.squeeze(0)
            R = R.squeeze(0)
            D = D.squeeze(0)
            H_phys = H_phys.squeeze(0)
            
        return {
            'S': S,
            'R': R,
            'D': D,
            'H_physics': H_phys
        }
