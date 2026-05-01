"""
MurrayLab CTX Orbital Tiles Loader.
Simple PyTorch Dataset for loading CTX 512x512 grayscale tiles.

Key features:
    - Loads grayscale tiles (already 512x512, no resize needed)
    - Quality check: rejects tiles with >30% near-saturated pixels
    - Normalizes to [0, 1] per-tile min-max
    - Replicates to 3 channels for CNN compatibility
    - No labels (qualitative demo dataset)
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.preprocessing.normalize import per_tile_minmax
from src.utils.io import load_image_grayscale


class CTXDataset(Dataset):
    """
    PyTorch Dataset for MurrayLab CTX orbital tiles.
    
    Each sample returns:
        image:    torch.Tensor (3, 512, 512), float32, normalized [0, 1]
        metadata: dict with filename, quality stats
    
    No labels — this dataset is used for qualitative demo and
    cross-domain evaluation only.
    """
    
    def __init__(
        self,
        tile_dirs,
        normalize_eps=1e-8,
        replicate_channels=3,
        quality_filter=True,
        max_saturated_fraction=0.30,
        saturation_threshold=0.05,
        max_tiles=None,
    ):
        """
        Args:
            tile_dirs: List of paths to tile directories (set 1, set 2).
            normalize_eps: Epsilon for normalization.
            replicate_channels: Number of output channels.
            quality_filter: If True, reject low-quality tiles.
            max_saturated_fraction: Max fraction of near-saturated pixels.
            saturation_threshold: Fraction of range to consider "saturated".
            max_tiles: Maximum number of tiles to include (None = all).
        """
        if isinstance(tile_dirs, (str, Path)):
            tile_dirs = [tile_dirs]
        
        self.tile_dirs = [Path(d) for d in tile_dirs]
        self.normalize_eps = normalize_eps
        self.replicate_channels = replicate_channels
        self.quality_filter = quality_filter
        self.max_saturated_fraction = max_saturated_fraction
        self.saturation_threshold = saturation_threshold
        
        # Build sample list
        self.samples = self._build_sample_list(max_tiles)
    
    def _build_sample_list(self, max_tiles):
        """Collect all tile paths from directories."""
        all_paths = []
        for tile_dir in self.tile_dirs:
            if not tile_dir.exists():
                continue
            for p in sorted(tile_dir.iterdir()):
                if p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.tif', '.tiff'):
                    all_paths.append(p)
        
        if max_tiles is not None:
            all_paths = all_paths[:max_tiles]
        
        return all_paths
    
    @staticmethod
    def compute_quality_stats(image):
        """
        Compute quality statistics for a tile.
        
        Args:
            image: numpy array (H, W), float32, raw values (0-255).
            
        Returns:
            dict with saturation stats.
        """
        imin, imax = image.min(), image.max()
        irange = imax - imin
        
        if irange < 1e-6:
            # Completely uniform tile
            return {
                "saturated_fraction": 1.0,
                "dynamic_range": 0.0,
                "mean_intensity": float(image.mean()),
                "passed": False,
            }
        
        # Near-saturated: within threshold% of min or max
        threshold = irange * 0.05  # 5% of range
        near_min = (image < imin + threshold).sum()
        near_max = (image > imax - threshold).sum()
        total = image.size
        saturated_fraction = (near_min + near_max) / total
        
        return {
            "saturated_fraction": float(saturated_fraction),
            "dynamic_range": float(irange),
            "mean_intensity": float(image.mean()),
        }
    
    def _passes_quality(self, image):
        """Check if a tile passes quality filter."""
        if not self.quality_filter:
            return True, {}
        
        stats = self.compute_quality_stats(image)
        passed = stats["saturated_fraction"] <= self.max_saturated_fraction
        stats["passed"] = passed
        return passed, stats
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path = self.samples[idx]
        
        # Load grayscale tile
        image = load_image_grayscale(path)
        
        # Quality stats
        _, quality_stats = self._passes_quality(image)
        
        # Normalize to [0, 1]
        image = per_tile_minmax(image, eps=self.normalize_eps)
        
        # Replicate to 3 channels
        if self.replicate_channels == 3 and image.ndim == 2:
            image = np.stack([image, image, image], axis=0)
        else:
            image = image[np.newaxis, ...]
        
        image_tensor = torch.from_numpy(image).float()
        
        metadata = {
            "filename": path.name,
            "source_dir": path.parent.name,
            "quality": quality_stats,
        }
        
        return image_tensor, metadata
    
    def get_quality_filtered_indices(self):
        """
        Pre-scan all tiles and return indices that pass quality filter.
        This is slow (loads every tile) but useful for one-time filtering.
        """
        passed_indices = []
        failed_indices = []
        
        for idx in range(len(self.samples)):
            image = load_image_grayscale(self.samples[idx])
            passed, stats = self._passes_quality(image)
            if passed:
                passed_indices.append(idx)
            else:
                failed_indices.append(idx)
        
        return passed_indices, failed_indices
    
    def select_demo_tiles(self, n=5, seed=42):
        """
        Select diverse demo tiles based on intensity distribution.
        Picks tiles spanning the range of mean intensities.
        
        Args:
            n: Number of tiles to select.
            seed: Random seed for reproducibility.
            
        Returns:
            List of indices.
        """
        rng = np.random.RandomState(seed)
        
        # Sample a subset to compute stats (avoid loading all 17K tiles)
        sample_size = min(500, len(self.samples))
        candidate_indices = rng.choice(len(self.samples), sample_size, replace=False)
        
        stats_list = []
        for idx in candidate_indices:
            image = load_image_grayscale(self.samples[idx])
            passed, stats = self._passes_quality(image)
            if passed:
                stats_list.append((idx, stats))
        
        if len(stats_list) < n:
            return [s[0] for s in stats_list]
        
        # Sort by mean intensity and pick evenly spaced
        stats_list.sort(key=lambda x: x[1]["mean_intensity"])
        step = len(stats_list) // n
        selected = [stats_list[i * step][0] for i in range(n)]
        
        return selected
    
    @classmethod
    def from_config(cls, base_cfg, dataset_cfg, max_tiles=None):
        """
        Create dataset from config objects.
        
        Args:
            base_cfg: Base project config.
            dataset_cfg: CTX dataset config.
            max_tiles: Limit number of tiles.
        """
        paths = base_cfg.paths
        preproc = dataset_cfg.preprocessing
        qf = dataset_cfg.quality_filter
        
        tile_dirs = [paths.murraylab.tiles_1, paths.murraylab.tiles_2]
        
        return cls(
            tile_dirs=tile_dirs,
            normalize_eps=preproc.normalize_eps,
            replicate_channels=preproc.replicate_channels,
            quality_filter=qf.enabled,
            max_saturated_fraction=qf.max_saturated_fraction,
            saturation_threshold=qf.saturation_threshold,
            max_tiles=max_tiles,
        )
