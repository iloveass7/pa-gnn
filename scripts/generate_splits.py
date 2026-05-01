"""
Generate train/val/test splits for AI4Mars dataset.
- Train: 70% of crowdsourced labels, stratified by dominant terrain class
- Val: 15% of crowdsourced labels, stratified
- Test: predefined expert labels (masked-gold-min3-100agree)

Split lists are saved as text files in data/splits/.
"""

import sys
import os
from pathlib import Path
from collections import Counter

import numpy as np
from PIL import Image

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import load_config, merge_configs
from src.utils.seed import set_seed
from src.data.transforms.label_remap import AI4MarsLabelRemapper


def get_dominant_class(label_path, remapper):
    """Get the dominant terrain class for a label mask."""
    label = np.array(Image.open(label_path), dtype=np.uint8)
    return remapper.get_dominant_class(label)


def stratified_split(stems, labels, train_ratio, val_ratio, seed=42):
    """
    Stratified split of stems into train/val sets.
    
    Args:
        stems: list of file stems
        labels: list of corresponding class labels (for stratification)
        train_ratio: fraction for train (e.g. 0.70)
        val_ratio: fraction for val (e.g. 0.15)
        seed: random seed
        
    Returns:
        (train_stems, val_stems)
    """
    rng = np.random.RandomState(seed)
    
    # Group by class
    class_groups = {}
    for stem, label in zip(stems, labels):
        if label not in class_groups:
            class_groups[label] = []
        class_groups[label].append(stem)
    
    train_stems = []
    val_stems = []
    
    # Adjust ratio: val is relative to (train+val)
    # train_ratio / (train_ratio + val_ratio) of non-test data goes to train
    total_ratio = train_ratio + val_ratio
    train_frac = train_ratio / total_ratio
    
    for cls, group_stems in sorted(class_groups.items()):
        shuffled = list(group_stems)
        rng.shuffle(shuffled)
        
        n = len(shuffled)
        n_train = int(round(n * train_frac))
        
        train_stems.extend(shuffled[:n_train])
        val_stems.extend(shuffled[n_train:])
    
    # Final shuffle within each split
    rng.shuffle(train_stems)
    rng.shuffle(val_stems)
    
    return train_stems, val_stems


def main():
    print("=" * 60)
    print("AI4Mars Split Generation")
    print("=" * 60)
    
    # Load configs
    base_cfg = load_config(project_root / "configs" / "base.yaml")
    ds_cfg = load_config(project_root / "configs" / "datasets" / "ai4mars.yaml")
    
    set_seed(base_cfg.project.seed)
    
    # Setup paths
    train_label_dir = Path(base_cfg.paths.ai4mars.labels_train)
    test_label_dir = Path(base_cfg.paths.ai4mars.labels_test)
    image_dir = Path(base_cfg.paths.ai4mars.images)
    splits_dir = Path(base_cfg.paths.splits)
    splits_dir.mkdir(parents=True, exist_ok=True)
    
    # Remapper for getting dominant class
    remapper = AI4MarsLabelRemapper.from_config(ds_cfg)
    
    # --- Crowdsourced labels: get all stems with matching images ---
    print("\nScanning crowdsourced labels...")
    label_files = sorted(train_label_dir.glob("*.png"))
    image_stems = set(p.stem for p in image_dir.iterdir() if p.suffix.upper() == '.JPG')
    
    valid_stems = []
    for lf in label_files:
        if lf.stem in image_stems:
            valid_stems.append(lf.stem)
    
    print(f"  Total labels: {len(label_files)}")
    print(f"  With matching images: {len(valid_stems)}")
    
    # --- Compute dominant class for stratification ---
    print("\nComputing dominant classes for stratification...")
    dominant_classes = []
    for i, stem in enumerate(valid_stems):
        lbl_path = train_label_dir / f"{stem}.png"
        dom_cls = get_dominant_class(lbl_path, remapper)
        dominant_classes.append(dom_cls)
        
        if (i + 1) % 2000 == 0:
            print(f"  Processed {i+1}/{len(valid_stems)}")
    
    # Class distribution before split
    class_counts = Counter(dominant_classes)
    print("\nClass distribution (crowdsourced labels):")
    class_names = {0: "soil", 1: "bedrock", 2: "sand", 3: "big_rock", -1: "all_null"}
    for cls_id in sorted(class_counts.keys()):
        count = class_counts[cls_id]
        pct = 100.0 * count / len(dominant_classes)
        name = class_names.get(cls_id, f"unknown_{cls_id}")
        print(f"  Class {cls_id} ({name}): {count} ({pct:.1f}%)")
    
    # --- Stratified split ---
    train_ratio = ds_cfg.splits.train_ratio
    val_ratio = ds_cfg.splits.val_ratio
    
    print(f"\nSplitting: train={train_ratio:.0%}, val={val_ratio:.0%}")
    train_stems, val_stems = stratified_split(
        valid_stems, dominant_classes,
        train_ratio, val_ratio,
        seed=base_cfg.project.seed
    )
    
    # --- Test set: expert labels ---
    test_stems = []
    for lf in sorted(test_label_dir.glob("*.png")):
        # Test labels have "_merged" suffix: remove it to get image stem
        stem = lf.stem.replace("_merged", "")
        if stem + ".JPG" in set(os.listdir(image_dir)):
            test_stems.append(stem)
    
    # --- Save split files ---
    for split_name, stems in [("train", train_stems), ("val", val_stems), ("test", test_stems)]:
        out_path = splits_dir / f"ai4mars_{split_name}.txt"
        with open(out_path, 'w') as f:
            for s in stems:
                f.write(s + "\n")
        print(f"\n  Saved {split_name}: {len(stems)} samples -> {out_path}")
    
    # --- Verify split integrity ---
    print("\n" + "=" * 60)
    print("Split Summary")
    print("=" * 60)
    print(f"  Train: {len(train_stems)} samples")
    print(f"  Val:   {len(val_stems)} samples")
    print(f"  Test:  {len(test_stems)} samples (expert gold-standard)")
    print(f"  Total: {len(train_stems) + len(val_stems) + len(test_stems)}")
    
    # Verify no overlap
    train_set = set(train_stems)
    val_set = set(val_stems)
    test_set = set(test_stems)
    
    assert len(train_set & val_set) == 0, "Train/val overlap!"
    assert len(train_set & test_set) == 0, "Train/test overlap!"
    assert len(val_set & test_set) == 0, "Val/test overlap!"
    print("\n  No overlaps between splits [OK]")
    
    # Per-split class distribution
    for split_name, stems in [("train", train_stems), ("val", val_stems)]:
        split_classes = []
        for s in stems:
            idx = valid_stems.index(s) if s in valid_stems else -1
            if idx >= 0:
                split_classes.append(dominant_classes[idx])
        
        counts = Counter(split_classes)
        print(f"\n  {split_name} class distribution:")
        for cls_id in sorted(counts.keys()):
            count = counts[cls_id]
            pct = 100.0 * count / len(split_classes)
            name = class_names.get(cls_id, f"unknown_{cls_id}")
            print(f"    Class {cls_id} ({name}): {count} ({pct:.1f}%)")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
