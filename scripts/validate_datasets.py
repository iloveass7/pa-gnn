"""
Dataset Validation Script
Checks that all dataset paths exist, counts images/labels, and reports basic statistics.
Run this after setting up the project to verify data accessibility.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config
from src.utils.io import list_files


def validate_directory(path, description, extensions=None):
    """Check if a directory exists and count files."""
    path = Path(path)
    if not path.exists():
        print(f"  [FAIL] {description}: {path} does NOT exist")
        return 0
    
    if not path.is_dir():
        print(f"  [FAIL] {description}: {path} is not a directory")
        return 0
    
    files = list_files(path, extensions=extensions)
    count = len(files)
    print(f"  [OK]   {description}: {count} files found at {path}")
    return count


def validate_file(path, description):
    """Check if a file exists."""
    path = Path(path)
    if not path.exists():
        print(f"  [FAIL] {description}: {path} does NOT exist")
        return False
    
    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"  [OK]   {description}: exists ({size_mb:.2f} MB)")
    return True


def main():
    config_path = PROJECT_ROOT / "configs" / "base.yaml"
    print(f"Loading config from: {config_path}\n")
    cfg = load_config(config_path)
    
    total_ok = 0
    total_fail = 0
    
    # === AI4Mars Dataset ===
    print("=" * 60)
    print("DATASET 1: AI4Mars (MSL NavCam)")
    print("=" * 60)
    
    # Images (EDR subdirectory)
    img_count = validate_directory(
        cfg.paths.ai4mars.images,
        "NavCam EDR images",
        extensions=['.jpg', '.jpeg', '.JPG', '.JPEG']
    )
    if img_count > 0:
        total_ok += 1
    else:
        total_fail += 1
    
    # Train labels
    train_count = validate_directory(
        cfg.paths.ai4mars.labels_train,
        "Train labels",
        extensions=['.png', '.PNG']
    )
    if train_count > 0:
        total_ok += 1
    else:
        total_fail += 1
    
    # Test labels (min3)
    test_count = validate_directory(
        cfg.paths.ai4mars.labels_test,
        "Test labels (min3-100agree)",
        extensions=['.png', '.PNG']
    )
    if test_count > 0:
        total_ok += 1
    else:
        total_fail += 1
    
    # Range masks
    if hasattr(cfg.paths.ai4mars, 'range_masks'):
        range_count = validate_directory(
            cfg.paths.ai4mars.range_masks,
            "Range masks (30m)",
            extensions=['.png', '.PNG', '.jpg', '.JPG']
        )
    
    print(f"\n  Summary: {img_count} images, {train_count} train labels, {test_count} test labels")
    
    # === MurrayLab CTX Dataset ===
    print("\n" + "=" * 60)
    print("DATASET 2: MurrayLab CTX Orbital Tiles")
    print("=" * 60)
    
    tiles1_count = validate_directory(
        cfg.paths.murraylab.tiles_1,
        "CTX tiles set 1",
        extensions=['.png', '.PNG']
    )
    
    tiles2_count = validate_directory(
        cfg.paths.murraylab.tiles_2,
        "CTX tiles set 2",
        extensions=['.png', '.PNG']
    )
    
    ctx_total = tiles1_count + tiles2_count
    if ctx_total > 0:
        total_ok += 1
    else:
        total_fail += 1
    
    print(f"\n  Summary: {ctx_total} total CTX tiles ({tiles1_count} + {tiles2_count})")
    
    # === HiRISE v3 Dataset ===
    print("\n" + "=" * 60)
    print("DATASET 3: HiRISE Map-Proj-v3")
    print("=" * 60)
    
    hirise_count = validate_directory(
        cfg.paths.hirise.images,
        "HiRISE crops",
        extensions=['.jpg', '.jpeg', '.JPG', '.JPEG']
    )
    if hirise_count > 0:
        total_ok += 1
    else:
        total_fail += 1
    
    validate_file(cfg.paths.hirise.labels, "Labels file")
    validate_file(cfg.paths.hirise.classmap, "Classmap CSV")
    
    # Read classmap to verify classes
    classmap_path = Path(cfg.paths.hirise.classmap)
    if classmap_path.exists():
        with open(classmap_path, 'r') as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
        print(f"\n  HiRISE classes found ({len(lines)}):")
        for line in lines:
            parts = line.split(',')
            if len(parts) >= 2:
                print(f"    ID {parts[0]}: {parts[1]}")
    
    # Count labels
    labels_path = Path(cfg.paths.hirise.labels)
    if labels_path.exists():
        with open(labels_path, 'r') as f:
            label_count = sum(1 for _ in f)
        print(f"\n  Labels file: {label_count} entries")
        
        estimated_originals = label_count // 7
        print(f"  Estimated original crops: ~{estimated_originals}")
    
    print(f"\n  Summary: {hirise_count} total HiRISE crops")
    
    # === Overall Summary ===
    print("\n" + "=" * 60)
    print("OVERALL VALIDATION SUMMARY")
    print("=" * 60)
    print(f"  Checks passed: {total_ok}")
    print(f"  Checks failed: {total_fail}")
    
    if total_fail == 0:
        print("\n  [PASS] All datasets validated successfully!")
    else:
        print(f"\n  [WARN] {total_fail} check(s) failed -- review paths above")
    
    return total_fail == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
