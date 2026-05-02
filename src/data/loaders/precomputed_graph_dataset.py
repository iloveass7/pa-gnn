"""
precomputed_graph_dataset.py — PyG Dataset that loads pre-saved .pt graph files.

Drop-in replacement for the on-the-fly graph building path in train_gnn.py.
Graphs are saved by scripts/precompute_graphs.py and loaded here in ~0.01s
instead of being rebuilt from scratch in ~0.8s per image.

Usage:
    from src.data.loaders.precomputed_graph_dataset import PrecomputedGraphDataset
    from torch_geometric.loader import DataLoader as PyGDataLoader

    train_ds = PrecomputedGraphDataset("d:/Mars/pa-gnn/data/processed/graphs/train")
    train_loader = PyGDataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)
"""

import json
from pathlib import Path

import torch
from torch_geometric.data import Dataset, Data


class PrecomputedGraphDataset(Dataset):
    """
    Loads pre-computed PyG graphs from .pt files on disk.

    Each .pt file contains a torch_geometric.data.Data object with:
        data.x           (N, 14)  — node feature matrix
        data.edge_index  (2, E)   — COO edge connectivity
        data.edge_attr   (E,)     — edge weights
        data.pos         (N, 2)   — centroid (y, x) positions
        data.y           (N,)     — node-level risk targets (or None)
        data.active_mask (N,)     — boolean active node mask
        data.label_map   (H, W)   — superpixel label image (long tensor)

    Args:
        root_dir:   path to a split directory created by precompute_graphs.py
                    (must contain manifest.json and graph_XXXXX.pt files)
        transform:  optional PyG transform applied on-the-fly after loading
    """

    def __init__(self, root_dir, transform=None):
        self.root_dir  = Path(root_dir)
        self.transform = transform

        manifest_path = self.root_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"manifest.json not found in {root_dir}. "
                f"Run scripts/precompute_graphs.py first."
            )

        with open(manifest_path) as f:
            self.manifest = json.load(f)

        # Pre-build sorted list of absolute file paths for fast __getitem__
        self._files = [
            self.root_dir / entry["file"]
            for entry in self.manifest
        ]
        self._stems = [entry["stem"] for entry in self.manifest]

    # ── PyG Dataset interface ──────────────────────────────────────────────

    def len(self):
        return len(self._files)

    def get(self, idx):
        data = torch.load(self._files[idx], map_location="cpu", weights_only=False)

        # Guarantee the expected attributes exist (backward-compat with older .pt)
        if not hasattr(data, "active_mask"):
            N = data.x.size(0)
            data.active_mask = torch.ones(N, dtype=torch.bool)

        # Ensure data.y is always a float tensor (never None) so batching works.
        # Graphs with no ground-truth get a tensor of -1 (all-ignore).
        if not hasattr(data, "y") or data.y is None:
            N = data.x.size(0)
            data.y = torch.full((N,), -1.0, dtype=torch.float32)

        # NOTE: do NOT attach data.stem — PyG DataLoader cannot collate strings
        # and will crash on batch_size > 1. Use dataset.stems[idx] if needed.

        # Drop label_map from batched data — it is a 2-D (H,W) tensor that
        # PyG would incorrectly batch along dim-0. It is never used in training.
        if hasattr(data, "label_map"):
            del data.label_map

        if self.transform is not None:
            data = self.transform(data)

        return data

    # ── Convenience ────────────────────────────────────────────────────────

    def __repr__(self):
        return (f"PrecomputedGraphDataset("
                f"root={self.root_dir}, "
                f"n_graphs={len(self)})")

    @property
    def stems(self):
        """Returns the list of image stem names (for debugging/verification)."""
        return self._stems

    @classmethod
    def from_split_dir(cls, graphs_base_dir, split, transform=None):
        """
        Convenience constructor.

        Args:
            graphs_base_dir: root output dir from precompute_graphs.py
            split: 'train' | 'val' | 'test'
            transform: optional PyG transform
        """
        split_dir = Path(graphs_base_dir) / split
        return cls(split_dir, transform=transform)
