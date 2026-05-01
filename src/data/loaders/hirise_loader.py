"""
HiRISE v3 Dataset Loader.
PyTorch Dataset class for loading HiRISE map-projected image crops with
image-level landmark labels.

Key features:
    - Loads grayscale 227x227 crops → resize to 512 → replicate to 3-ch
    - Filters to originals only (augmentation index 0, no aug suffix)
    - Loads image-level labels from labels txt file + classmap CSV
    - Converts landmark class to continuous risk score
"""

import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.preprocessing.normalize import per_tile_minmax
from src.data.preprocessing.resize import resize_image
from src.data.transforms.label_remap import HiRISELabelRemapper
from src.utils.io import load_image_grayscale


# Suffixes that identify augmented (non-original) crops
_AUG_SUFFIXES = ("-r90", "-r180", "-r270", "-fh", "-fv", "-brt")


class HiRISEDataset(Dataset):
    """
    PyTorch Dataset for HiRISE v3 landmark classification crops.
    
    Each sample returns:
        image:      torch.Tensor (3, 512, 512), float32, normalized [0, 1]
        risk_score: torch.Tensor (1,), float32, risk score for entire image
        metadata:   dict with filename, class index, class name
    """
    
    def __init__(
        self,
        image_dir,
        labels_file,
        classmap_file,
        target_size=512,
        normalize_eps=1e-8,
        use_originals_only=True,
        label_remapper=None,
        replicate_channels=3,
        split_file=None,
    ):
        """
        Args:
            image_dir: Path to HiRISE image crops directory.
            labels_file: Path to labels-map-proj-v3.txt.
            classmap_file: Path to landmarks classmap CSV.
            target_size: Resize target (227→512).
            normalize_eps: Epsilon for normalization.
            use_originals_only: If True, filter out augmented crops.
            label_remapper: HiRISELabelRemapper instance.
            replicate_channels: Number of output channels.
            split_file: Optional path to text file with selected stems.
        """
        self.image_dir = Path(image_dir)
        self.target_size = target_size
        self.normalize_eps = normalize_eps
        self.replicate_channels = replicate_channels
        
        # Load classmap
        self.classmap = self._load_classmap(classmap_file)
        
        # Load labels (filename → class index)
        self.label_dict = self._load_labels(labels_file)
        
        # Setup remapper
        self.remapper = label_remapper or HiRISELabelRemapper()
        
        # Build sample list
        self.samples = self._build_sample_list(
            use_originals_only, split_file
        )
    
    def _load_classmap(self, classmap_file):
        """Load class index → class name mapping from CSV."""
        classmap = {}
        with open(classmap_file, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2:
                    classmap[int(row[0])] = row[1].strip()
        return classmap
    
    def _load_labels(self, labels_file):
        """Load filename → class index mapping from labels txt."""
        label_dict = {}
        with open(labels_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.rsplit(' ', 1)
                if len(parts) == 2:
                    filename = parts[0].strip()
                    class_idx = int(parts[1])
                    label_dict[filename] = class_idx
        return label_dict
    
    def _is_original(self, filename):
        """Check if a filename is an original (not augmented) crop."""
        stem = Path(filename).stem
        return not any(stem.endswith(suffix) for suffix in _AUG_SUFFIXES)
    
    def _build_sample_list(self, use_originals_only, split_file):
        """Build list of valid samples (image exists + has label)."""
        # Get allowed stems from split file if provided
        allowed_stems = None
        if split_file is not None:
            with open(split_file, 'r') as f:
                allowed_stems = set(line.strip() for line in f if line.strip())
        
        samples = []
        
        # Iterate over images that exist on disk AND have labels
        for img_path in sorted(self.image_dir.iterdir()):
            if img_path.suffix.lower() not in ('.jpg', '.jpeg', '.png'):
                continue
            
            filename = img_path.name
            
            # Filter originals only
            if use_originals_only and not self._is_original(filename):
                continue
            
            # Must have a label
            if filename not in self.label_dict:
                continue
            
            # Check split file
            if allowed_stems is not None:
                if img_path.stem not in allowed_stems:
                    continue
            
            class_idx = self.label_dict[filename]
            
            samples.append({
                "path": img_path,
                "filename": filename,
                "class_index": class_idx,
                "class_name": self.classmap.get(class_idx, f"unknown_{class_idx}"),
            })
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load grayscale image (227x227)
        image = load_image_grayscale(sample["path"])
        
        # Resize to target (512x512)
        image = resize_image(image / 255.0, self.target_size, "bilinear")
        
        # Normalize to [0, 1]
        image = per_tile_minmax(image, eps=self.normalize_eps)
        
        # Replicate to 3 channels
        if self.replicate_channels == 3 and image.ndim == 2:
            image = np.stack([image, image, image], axis=0)
        else:
            image = image[np.newaxis, ...]
        
        # Get risk score for this class
        risk_score = self.remapper(sample["class_index"])
        
        # Convert to tensors
        image_tensor = torch.from_numpy(image).float()
        risk_tensor = torch.tensor([risk_score], dtype=torch.float32)
        
        metadata = {
            "filename": sample["filename"],
            "class_index": sample["class_index"],
            "class_name": sample["class_name"],
            "risk_score": risk_score,
            "risk_category": self.remapper.get_risk_category(
                sample["class_index"]
            ),
        }
        
        return image_tensor, risk_tensor, metadata
    
    @classmethod
    def from_config(cls, base_cfg, dataset_cfg, split_file=None):
        """
        Create dataset from config objects.
        
        Args:
            base_cfg: Base project config.
            dataset_cfg: HiRISE dataset config.
            split_file: Optional split file path.
        """
        paths = base_cfg.paths
        preproc = dataset_cfg.preprocessing
        eval_cfg = dataset_cfg.evaluation
        
        remapper = HiRISELabelRemapper.from_config(dataset_cfg)
        
        return cls(
            image_dir=paths.hirise.images,
            labels_file=paths.hirise.labels,
            classmap_file=paths.hirise.classmap,
            target_size=preproc.target_size,
            normalize_eps=preproc.normalize_eps,
            use_originals_only=eval_cfg.use_originals_only,
            label_remapper=remapper,
            replicate_channels=preproc.replicate_channels,
            split_file=split_file,
        )
    
    def get_class_distribution(self):
        """Compute class distribution across all samples."""
        counts = {}
        for s in self.samples:
            cls_name = s["class_name"]
            counts[cls_name] = counts.get(cls_name, 0) + 1
        return counts
