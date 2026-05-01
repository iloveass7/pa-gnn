"""
Stage 2 Verification Script.
Validates physics feature extraction on AI4Mars and CTX images.

Produces:
    - Side-by-side grids for 5 AI4Mars images + 3 CTX tiles.
      Format: Original | S | R | D | H_physics
    - Performance benchmark (computation time per tile)
"""

import sys
import time
from pathlib import Path
import random

import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add project root
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import load_config
from src.utils.seed import set_seed, get_device
from src.data.loaders.ai4mars_loader import AI4MarsDataset
from src.data.loaders.ctx_loader import CTXDataset
from src.data.transforms.physics_features import PhysicsFeatureExtractor

def visualize_physics_grid(images_list, extractor, device, out_path, title_prefix="Physics Features"):
    """
    images_list: list of tuples (img_tensor, name)
                 img_tensor is (3, H, W)
    """
    num_imgs = len(images_list)
    fig, axes = plt.subplots(num_imgs, 5, figsize=(20, 4 * num_imgs))
    if num_imgs == 1:
        axes = [axes]
        
    for i, (img, name) in enumerate(images_list):
        # Compute physics
        img_device = img.to(device)
        t0 = time.perf_counter()
        phys_dict = extractor(img_device)
        t_ms = (time.perf_counter() - t0) * 1000.0
        
        S = phys_dict['S'].cpu().squeeze().numpy()
        R = phys_dict['R'].cpu().squeeze().numpy()
        D = phys_dict['D'].cpu().squeeze().numpy()
        H = phys_dict['H_physics'].cpu().squeeze().numpy()
        
        orig_img = img.mean(dim=0).numpy() # Grayscale for display
        
        axes[i][0].imshow(orig_img, cmap='gray')
        axes[i][0].set_title(f"{name}\n({t_ms:.1f} ms)", fontsize=10)
        axes[i][0].axis('off')
        
        axes[i][1].imshow(S, cmap='viridis', vmin=0, vmax=1)
        axes[i][1].set_title("Slope (S)", fontsize=10)
        axes[i][1].axis('off')
        
        axes[i][2].imshow(R, cmap='viridis', vmin=0, vmax=1)
        axes[i][2].set_title("Roughness (R)", fontsize=10)
        axes[i][2].axis('off')
        
        axes[i][3].imshow(D, cmap='viridis', vmin=0, vmax=1)
        axes[i][3].set_title("Discontinuity (D)", fontsize=10)
        axes[i][3].axis('off')
        
        axes[i][4].imshow(H, cmap='magma', vmin=0, vmax=1)
        axes[i][4].set_title("H_physics", fontsize=10)
        axes[i][4].axis('off')

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")

def main():
    print("=" * 60)
    print("STAGE 2 VERIFICATION")
    print("=" * 60)
    
    base_cfg = load_config(project_root / "configs" / "base.yaml")
    ai4mars_cfg = load_config(project_root / "configs" / "datasets" / "ai4mars.yaml")
    ctx_cfg = load_config(project_root / "configs" / "datasets" / "ctx.yaml")
    phys_cfg = load_config(project_root / "configs" / "physics.yaml")
    
    set_seed(base_cfg.project.seed)
    device = get_device(base_cfg.project.device)
    
    results_dir = Path(base_cfg.paths.results) / "stage2"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    extractor = PhysicsFeatureExtractor.from_config(phys_cfg).to(device)
    
    # 1. AI4Mars
    print("\nLoading AI4Mars val dataset...")
    val_ds = AI4MarsDataset.from_config(base_cfg, ai4mars_cfg, split="val")
    # Pick 5 random
    indices = random.sample(range(len(val_ds)), min(5, len(val_ds)))
    ai4mars_images = []
    for idx in indices:
        img, _, meta = val_ds[idx]
        ai4mars_images.append((img, f"AI4Mars #{idx}\n{meta['dominant_class']}"))
        
    print("Extracting AI4Mars physics features...")
    visualize_physics_grid(
        ai4mars_images, extractor, device, 
        results_dir / "ai4mars_physics_grid.png",
        title_prefix="AI4Mars Physics"
    )

    # 2. CTX
    print("\nLoading CTX dataset...")
    ctx_ds = CTXDataset.from_config(base_cfg, ctx_cfg, max_tiles=100)
    ctx_indices = ctx_ds.select_demo_tiles(n=3, seed=base_cfg.project.seed)
    ctx_images = []
    for idx in ctx_indices:
        img, meta = ctx_ds[idx]
        ctx_images.append((img, f"CTX {meta['filename'][:15]}"))
        
    print("Extracting CTX physics features...")
    visualize_physics_grid(
        ctx_images, extractor, device, 
        results_dir / "ctx_physics_grid.png",
        title_prefix="CTX Physics"
    )

    # 3. Benchmark
    print("\nRunning benchmark (100 iterations, batch size 1)...")
    img = ctx_images[0][0].unsqueeze(0).to(device) # (1, 3, 512, 512)
    # Warmup
    for _ in range(10):
        _ = extractor(img)
        
    t0 = time.perf_counter()
    for _ in range(100):
        _ = extractor(img)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    avg_ms = ((t1 - t0) / 100.0) * 1000.0
    print(f"Average computation time per tile: {avg_ms:.2f} ms")
    
    print("\n" + "=" * 60)
    print("STAGE 2 VERIFICATION COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
