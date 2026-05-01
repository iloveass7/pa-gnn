"""
AI4Mars MSL NavCam Dataset Loader.
PyTorch Dataset class for loading EDR images with NAV terrain labels.

Key features:
    - Loads grayscale EDR images (1024x1024) → resize to 512 → replicate to 3-ch
    - Applies range mask (30m) to exclude far-field pixels
    - Converts integer labels to continuous risk scores via label_remap
    - Supports train/val/test splits (from saved split files)
    - Training augmentations applied jointly to image+label
"""

import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.preprocessing.normalize import per_tile_minmax
from src.data.preprocessing.resize import resize_image, resize_label
from src.data.preprocessing.augmentations import JointAugmentation, NoAugmentation
from src.data.transforms.label_remap import AI4MarsLabelRemapper
from src.utils.io import load_image_grayscale, load_label_mask


class AI4MarsDataset(Dataset):
    """
    PyTorch Dataset for AI4Mars MSL NavCam EDR images with NAV terrain labels.
    
    Each sample returns:
        image:     torch.Tensor (3, H, W), float32, normalized [0, 1]
        risk_map:  torch.Tensor (1, H, W), float32, risk scores; -1 = ignore
        metadata:  dict with filename, dominant class, etc.
    """
    
    def __init__(
        self,
        image_dir,
        label_dir,
        range_mask_dir=None,
        split_file=None,
        file_list=None,
        image_size=512,
        normalize_eps=1e-8,
        label_remapper=None,
        augmentation=None,
        replicate_channels=3,
    ):
        """
        Args:
            image_dir: Path to EDR image directory.
            label_dir: Path to label PNG directory.
            range_mask_dir: Path to range mask directory (optional).
            split_file: Path to text file listing filenames for this split.
            file_list: Explicit list of stems (alternative to split_file).
            image_size: Target image size (default 512).
            normalize_eps: Epsilon for min-max normalization.
            label_remapper: AI4MarsLabelRemapper instance.
            augmentation: JointAugmentation or NoAugmentation instance.
            replicate_channels: Number of output channels (1→3 replication).
        """
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.range_mask_dir = Path(range_mask_dir) if range_mask_dir else None
        self.image_size = image_size
        self.normalize_eps = normalize_eps
        self.replicate_channels = replicate_channels
        
        self.remapper = label_remapper or AI4MarsLabelRemapper()
        self.augmentation = augmentation or NoAugmentation()
        
        # Build file list
        self.samples = self._build_sample_list(split_file, file_list)
    
    def _build_sample_list(self, split_file, file_list):
        """
        Build list of (image_path, label_path, range_mask_path) tuples.
        """
        if split_file is not None:
            # Load stems from split file
            with open(split_file, 'r') as f:
                stems = [line.strip() for line in f if line.strip()]
        elif file_list is not None:
            stems = file_list
        else:
            # Use all labels as reference (every label has a matching image)
            stems = sorted([
                p.stem for p in self.label_dir.iterdir()
                if p.suffix.lower() == '.png'
            ])
        
        samples = []
        for stem in stems:
            # Image is .JPG, label is .png
            img_path = self.image_dir / f"{stem}.JPG"
            lbl_path = self.label_dir / f"{stem}.png"
            
            if not img_path.exists():
                # Try lowercase extension
                img_path = self.image_dir / f"{stem}.jpg"
            
            # Test labels have _merged suffix: try that too
            if not lbl_path.exists():
                lbl_path = self.label_dir / f"{stem}_merged.png"
            
            if not img_path.exists() or not lbl_path.exists():
                continue
            
            # Range mask (optional)
            rng_path = None
            if self.range_mask_dir is not None:
                # Range mask naming: replace EDR with RNG
                rng_stem = stem.replace("EDR", "RNG")
                rng_path = self.range_mask_dir / f"{rng_stem}.png"
                if not rng_path.exists():
                    rng_path = None
            
            samples.append({
                "stem": stem,
                "image": img_path,
                "label": lbl_path,
                "range_mask": rng_path,
            })
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load grayscale image (H, W), float32, values 0-255
        image = load_image_grayscale(sample["image"])
        
        # Load label mask (H, W), uint8, values {0,1,2,3,255}
        label = load_label_mask(sample["label"])
        
        # Apply range mask: set out-of-range pixels to null (255)
        if sample["range_mask"] is not None:
            range_mask = load_label_mask(sample["range_mask"])
            # AI4Mars rng-30m mask: 0 (black) = within 30m (valid), non-zero = out of range
            label[range_mask != 0] = 255
        
        # Resize image (bilinear) and label (nearest)
        image = resize_image(image / 255.0, self.image_size, "bilinear")
        label = resize_label(label, self.image_size)
        
        # Normalize image to [0, 1] (per-tile min-max)
        image = per_tile_minmax(image, eps=self.normalize_eps)
        
        # Apply augmentations (jointly to image+label)
        image, label = self.augmentation(image, label)
        
        # Convert label to continuous risk scores
        risk_map = self.remapper(label)
        
        # Get dominant class (before remapping, for stratification metadata)
        dominant_class = self.remapper.get_dominant_class(
            resize_label(load_label_mask(sample["label"]), self.image_size)
        )
        
        # Replicate grayscale to 3-channel
        if self.replicate_channels == 3 and image.ndim == 2:
            image = np.stack([image, image, image], axis=0)  # (3, H, W)
        else:
            image = image[np.newaxis, ...]  # (1, H, W)
        
        # Risk map: (1, H, W)
        risk_map = risk_map[np.newaxis, ...]
        
        # Convert to tensors
        image_tensor = torch.from_numpy(image).float()
        risk_tensor = torch.from_numpy(risk_map).float()
        
        metadata = {
            "stem": sample["stem"],
            "dominant_class": dominant_class,
            "has_range_mask": sample["range_mask"] is not None,
        }
        
        return image_tensor, risk_tensor, metadata
    
    @classmethod
    def from_config(cls, base_cfg, dataset_cfg, split="train", split_dir=None):
        """
        Create dataset from config objects.
        
        Args:
            base_cfg: Base project config (paths, image settings).
            dataset_cfg: AI4Mars dataset config.
            split: "train", "val", or "test".
            split_dir: Directory containing split files. If None, uses base_cfg.
        """
        paths = base_cfg.paths
        preproc = dataset_cfg.preprocessing
        
        # Determine directories based on split
        if split == "test":
            label_dir = paths.ai4mars.labels_test
        else:
            label_dir = paths.ai4mars.labels_train
        
        # Split file
        split_file = None
        if split_dir is not None:
            sf = Path(split_dir) / f"ai4mars_{split}.txt"
            if sf.exists():
                split_file = sf
        elif hasattr(paths, 'splits'):
            sf = Path(paths.splits) / f"ai4mars_{split}.txt"
            if sf.exists():
                split_file = sf
        
        # Label remapper
        remapper = AI4MarsLabelRemapper.from_config(dataset_cfg)
        
        # Augmentation (only for training)
        if split == "train" and dataset_cfg.augmentation.enabled:
            augmentation = JointAugmentation.from_config(dataset_cfg)
        else:
            augmentation = NoAugmentation()
        
        return cls(
            image_dir=paths.ai4mars.images,
            label_dir=label_dir,
            range_mask_dir=paths.ai4mars.range_masks,
            split_file=split_file,
            image_size=preproc.image_size,
            normalize_eps=preproc.normalize_eps,
            label_remapper=remapper,
            augmentation=augmentation,
            replicate_channels=preproc.replicate_channels,
        )
