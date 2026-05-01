"""
Stage 1 Verification Script.
Validates all dataset loaders, label remapping, preprocessing, and splits.

Produces:
    - Sample image+label grid for AI4Mars (8 samples with risk colormap)
    - Class distribution histogram for AI4Mars train split
    - Sample HiRISE crops with remapped risk class
    - CTX demo tiles with quality stats
    - Split file counts
"""

import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from collections import Counter

# Add project root
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.data.loaders.ai4mars_loader import AI4MarsDataset
from src.data.loaders.hirise_loader import HiRISEDataset
from src.data.loaders.ctx_loader import CTXDataset
from src.data.transforms.label_remap import AI4MarsLabelRemapper, HiRISELabelRemapper


def verify_ai4mars(base_cfg, ds_cfg, results_dir):
    """Verify AI4Mars dataset loader and visualize samples."""
    print("\n" + "=" * 60)
    print("1. AI4Mars Dataset Verification")
    print("=" * 60)
    
    remapper = AI4MarsLabelRemapper.from_config(ds_cfg)
    splits_dir = Path(base_cfg.paths.splits)
    
    # Check splits exist
    for split in ["train", "val", "test"]:
        sf = splits_dir / f"ai4mars_{split}.txt"
        if sf.exists():
            with open(sf) as f:
                count = sum(1 for line in f if line.strip())
            print(f"  {split} split: {count} samples")
        else:
            print(f"  {split} split: NOT FOUND at {sf}")
    
    # Load train split
    print("\n  Loading train dataset...")
    t0 = time.time()
    train_ds = AI4MarsDataset.from_config(base_cfg, ds_cfg, split="train")
    print(f"  Loaded {len(train_ds)} train samples in {time.time()-t0:.1f}s")
    
    # Load val split
    val_ds = AI4MarsDataset.from_config(base_cfg, ds_cfg, split="val")
    print(f"  Loaded {len(val_ds)} val samples")
    
    # Load test split
    test_ds = AI4MarsDataset.from_config(base_cfg, ds_cfg, split="test")
    print(f"  Loaded {len(test_ds)} test samples")
    
    # --- Visualize 8 samples as grid ---
    print("\n  Generating sample grid...")
    fig, axes = plt.subplots(2, 8, figsize=(28, 7))
    
    # Risk colormap (blue=safe → red=hazardous)
    risk_cmap = plt.cm.RdYlGn_r  # Red=high risk, green=low risk
    
    indices = np.linspace(0, len(train_ds) - 1, 8, dtype=int)
    for col, idx in enumerate(indices):
        img, risk, meta = train_ds[idx]
        
        # Image (use first channel)
        axes[0, col].imshow(img[0].numpy(), cmap='gray')
        axes[0, col].set_title(f"#{idx}", fontsize=8)
        axes[0, col].axis('off')
        
        # Risk map
        risk_map = risk[0].numpy()
        # Mask ignore regions
        masked_risk = np.ma.masked_where(risk_map < 0, risk_map)
        axes[1, col].imshow(masked_risk, cmap=risk_cmap, vmin=0, vmax=1)
        axes[1, col].set_title(f"cls={meta['dominant_class']}", fontsize=8)
        axes[1, col].axis('off')
    
    axes[0, 0].set_ylabel("Image", fontsize=10)
    axes[1, 0].set_ylabel("Risk Map", fontsize=10)
    fig.suptitle("AI4Mars Train Samples — Image + Risk Colormap", fontsize=14)
    plt.tight_layout()
    
    out_path = results_dir / "ai4mars_sample_grid.png"
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")
    
    # --- Class distribution histogram ---
    print("\n  Computing class distribution for train split...")
    class_counts = Counter()
    for i in range(len(train_ds)):
        _, _, meta = train_ds[i]
        class_counts[meta['dominant_class']] += 1
        if (i + 1) % 2000 == 0:
            print(f"    Processed {i+1}/{len(train_ds)}")
    
    class_names = {0: "Soil", 1: "Bedrock", 2: "Sand", 3: "Big Rock", -1: "All Null"}
    labels = [class_names.get(k, f"Unk_{k}") for k in sorted(class_counts.keys())]
    values = [class_counts[k] for k in sorted(class_counts.keys())]
    colors = ['#2ecc71', '#e67e22', '#f1c40f', '#e74c3c', '#95a5a6']
    
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color=colors[:len(labels)], edgecolor='black', linewidth=0.5)
    ax.set_xlabel("Dominant Terrain Class", fontsize=12)
    ax.set_ylabel("Number of Samples", fontsize=12)
    ax.set_title("AI4Mars Train Split — Class Distribution", fontsize=14)
    
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 50,
                f'{val}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    out_path = results_dir / "ai4mars_class_distribution.png"
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")
    
    return class_counts


def verify_hirise(base_cfg, ds_cfg, results_dir):
    """Verify HiRISE dataset loader."""
    print("\n" + "=" * 60)
    print("2. HiRISE v3 Dataset Verification")
    print("=" * 60)
    
    print("  Loading HiRISE dataset (originals only)...")
    t0 = time.time()
    hirise_ds = HiRISEDataset.from_config(base_cfg, ds_cfg)
    print(f"  Loaded {len(hirise_ds)} original crops in {time.time()-t0:.1f}s")
    
    # Class distribution
    class_dist = hirise_ds.get_class_distribution()
    print("\n  Class distribution:")
    for cls_name, count in sorted(class_dist.items(), key=lambda x: -x[1]):
        pct = 100.0 * count / len(hirise_ds)
        print(f"    {cls_name}: {count} ({pct:.1f}%)")
    
    # --- Visualize sample crops ---
    print("\n  Generating sample crops...")
    # Pick one sample per class
    class_samples = {}
    for idx in range(len(hirise_ds)):
        img, risk, meta = hirise_ds[idx]
        cls_name = meta['class_name']
        if cls_name not in class_samples:
            class_samples[cls_name] = (img, risk, meta)
        if len(class_samples) == 8:
            break
    
    n_classes = len(class_samples)
    fig, axes = plt.subplots(1, n_classes, figsize=(3 * n_classes, 3.5))
    if n_classes == 1:
        axes = [axes]
    
    for i, (cls_name, (img, risk, meta)) in enumerate(sorted(class_samples.items())):
        axes[i].imshow(img[0].numpy(), cmap='gray')
        risk_val = meta['risk_score']
        category = meta['risk_category']
        color = {'safe': '#2ecc71', 'uncertain': '#f39c12', 'hazardous': '#e74c3c'}[category]
        axes[i].set_title(f"{cls_name}\nrisk={risk_val:.2f} ({category})",
                          fontsize=8, color=color, fontweight='bold')
        axes[i].axis('off')
    
    fig.suptitle("HiRISE v3 — Sample Crops by Class (with Risk Mapping)", fontsize=13)
    plt.tight_layout()
    out_path = results_dir / "hirise_sample_crops.png"
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")
    
    return class_dist


def verify_ctx(base_cfg, ds_cfg, results_dir):
    """Verify CTX dataset loader."""
    print("\n" + "=" * 60)
    print("3. CTX Tile Verification")
    print("=" * 60)
    
    print("  Loading CTX dataset...")
    t0 = time.time()
    ctx_ds = CTXDataset.from_config(base_cfg, ds_cfg, max_tiles=1000)
    print(f"  Loaded {len(ctx_ds)} tiles in {time.time()-t0:.1f}s")
    
    # Select demo tiles
    print("  Selecting demo tiles...")
    demo_indices = ctx_ds.select_demo_tiles(n=5, seed=42)
    print(f"  Selected {len(demo_indices)} demo tiles")
    
    # Visualize demo tiles
    fig, axes = plt.subplots(1, len(demo_indices), figsize=(3 * len(demo_indices), 3.5))
    if len(demo_indices) == 1:
        axes = [axes]
    
    for i, idx in enumerate(demo_indices):
        img, meta = ctx_ds[idx]
        quality = meta.get('quality', {})
        sat_frac = quality.get('saturated_fraction', 0)
        dyn_range = quality.get('dynamic_range', 0)
        
        axes[i].imshow(img[0].numpy(), cmap='gray')
        axes[i].set_title(
            f"{meta['filename'][:20]}...\nsat={sat_frac:.2f} rng={dyn_range:.0f}",
            fontsize=7
        )
        axes[i].axis('off')
    
    fig.suptitle("CTX Demo Tiles — with Quality Stats", fontsize=13)
    plt.tight_layout()
    out_path = results_dir / "ctx_demo_tiles.png"
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")
    
    return demo_indices


def verify_label_remap(results_dir):
    """Verify label remapping is correct."""
    print("\n" + "=" * 60)
    print("4. Label Remapping Verification")
    print("=" * 60)
    
    # AI4Mars
    remapper = AI4MarsLabelRemapper()
    test_mask = np.array([[0, 1, 2, 3, 255]], dtype=np.uint8)
    risk = remapper(test_mask)
    print("  AI4Mars remapping:")
    for val, name in [(0, "soil"), (1, "bedrock"), (2, "sand"), (3, "big_rock"), (255, "null")]:
        r = risk[0, list(test_mask[0]).index(val)]
        print(f"    {name} (px={val}) -> risk={r:.2f}")
    
    # HiRISE
    hi_remapper = HiRISELabelRemapper()
    print("\n  HiRISE remapping:")
    for idx in range(8):
        name = hi_remapper.class_name(idx)
        risk = hi_remapper(idx)
        cat = hi_remapper.get_risk_category(idx)
        print(f"    {name} (cls={idx}) -> risk={risk:.2f} ({cat})")


def main():
    print("=" * 60)
    print("STAGE 1 VERIFICATION")
    print("=" * 60)
    
    base_cfg = load_config(project_root / "configs" / "base.yaml")
    ai4mars_cfg = load_config(project_root / "configs" / "datasets" / "ai4mars.yaml")
    hirise_cfg = load_config(project_root / "configs" / "datasets" / "hirise.yaml")
    ctx_cfg = load_config(project_root / "configs" / "datasets" / "ctx.yaml")
    
    set_seed(base_cfg.project.seed)
    
    results_dir = Path(base_cfg.paths.results) / "stage1"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Label remapping verification
    verify_label_remap(results_dir)
    
    # 2. AI4Mars verification
    ai4mars_counts = verify_ai4mars(base_cfg, ai4mars_cfg, results_dir)
    
    # 3. HiRISE verification
    hirise_dist = verify_hirise(base_cfg, hirise_cfg, results_dir)
    
    # 4. CTX verification
    ctx_demos = verify_ctx(base_cfg, ctx_cfg, results_dir)
    
    print("\n" + "=" * 60)
    print("STAGE 1 VERIFICATION COMPLETE")
    print("=" * 60)
    print(f"\n  Results saved to: {results_dir}")
    print("  Files:")
    for f in sorted(results_dir.iterdir()):
        print(f"    - {f.name}")


if __name__ == "__main__":
    main()
