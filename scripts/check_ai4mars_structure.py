"""Quick check of AI4Mars image directory structure."""
from pathlib import Path

base = Path(r"d:\Mars\ai4mars-dataset-merged-0.6\ai4mars-dataset-merged-0.6\msl\ncam\images")
print(f"Base dir: {base}")
print(f"Subdirs: {[d.name for d in base.iterdir() if d.is_dir()]}")

edr = base / "edr"
if edr.exists():
    files = list(edr.iterdir())
    print(f"\nedr/ total items: {len(files)}")
    img_files = [f for f in files if f.suffix.upper() in ['.JPG', '.JPEG', '.PNG']]
    print(f"edr/ image files: {len(img_files)}")
    if img_files:
        print(f"Sample filename: {img_files[0].name}")

rng = base / "rng-30m"
if rng.exists():
    files = list(rng.iterdir())
    print(f"\nrng-30m/ total items: {len(files)}")
