import sys
import argparse
from pathlib import Path

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add project root
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import load_config
from src.utils.seed import set_seed, get_device
from src.data.loaders.ctx_loader import CTXDataset
from src.inference.pipeline import PA_GNN_Pipeline

def generate_visualizations(img_tensor, data, fusion_dict, paths, save_dir, prefix="ctx"):
    img_np = img_tensor.mean(dim=0).cpu().numpy()
    
    # 1. H_phys vs H_learn vs H_final
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(img_np, cmap='gray'); axes[0].set_title("Input"); axes[0].axis('off')
    axes[1].imshow(fusion_dict['h_physics'][0,0].cpu().numpy(), cmap=plt.cm.RdYlGn_r, vmin=0, vmax=1)
    axes[1].set_title("H_physics"); axes[1].axis('off')
    axes[2].imshow(fusion_dict['h_learned'][0,0].cpu().numpy(), cmap=plt.cm.RdYlGn_r, vmin=0, vmax=1)
    axes[2].set_title("H_learned"); axes[2].axis('off')
    axes[3].imshow(fusion_dict['h_final'][0,0].cpu().numpy(), cmap=plt.cm.RdYlGn_r, vmin=0, vmax=1)
    axes[3].set_title("H_final"); axes[3].axis('off')
    plt.tight_layout()
    plt.savefig(save_dir / f"{prefix}_risk_components.png", dpi=150)
    plt.close()
    
    # 2. Alpha map overlay
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img_np, cmap='gray')
    im = ax.imshow(fusion_dict['alpha'][0,0].cpu().numpy(), cmap='coolwarm', alpha=0.5, vmin=0, vmax=1)
    ax.set_title("Alpha Map (Red=CNN, Blue=Physics)")
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(save_dir / f"{prefix}_alpha_map.png", dpi=150)
    plt.close()
    
    # 3. Path Comparison (B1 vs Proposed)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax in axes:
        ax.imshow(img_np, cmap='gray')
        ax.axis('off')
        
    def plot_on_ax(ax, path_det, color, label):
        if not path_det: return
        xs = [p['pos'][1] for p in path_det]
        ys = [p['pos'][0] for p in path_det]
        ax.plot(xs, ys, color=color, linewidth=2, label=label)
        ax.plot(xs[0], ys[0], 'bo')
        ax.plot(xs[-1], ys[-1], 'go')
        
    plot_on_ax(axes[0], paths.get('b1_euclidean'), 'red', 'B1 (Euclidean)')
    axes[0].set_title("Baseline B1 Path")
    axes[0].legend()
    
    plot_on_ax(axes[1], paths.get('proposed'), 'blue', 'Proposed (PA-GNN)')
    axes[1].set_title("Proposed PA-GNN Path")
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(save_dir / f"{prefix}_path_comparison.png", dpi=150)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    args, _ = parser.parse_known_args()
    
    base_cfg = load_config('configs/base.yaml')
    cnn_cfg = load_config('configs/cnn/mobilenetv3.yaml')
    phys_cfg = load_config('configs/physics.yaml')
    fusion_cfg = load_config('configs/fusion/adaptive_fusion.yaml')
    gat_cfg = load_config('configs/gnn/gatv2.yaml')
    ctx_cfg = load_config('configs/datasets/ctx.yaml')
    
    device = get_device(base_cfg.project.device)
    set_seed(base_cfg.project.seed)
    
    results_dir = Path(base_cfg.paths.results) / "stage7_ctx"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    pipeline = PA_GNN_Pipeline(base_cfg, cnn_cfg, phys_cfg, fusion_cfg, gat_cfg, device)
    
    ctx_ds = CTXDataset.from_config(base_cfg, ctx_cfg, max_tiles=100)
    indices = ctx_ds.select_demo_tiles(n=3, seed=base_cfg.project.seed)
    
    print("Running Demo on CTX Tiles...")
    for idx in indices:
        img_tensor, meta = ctx_ds[idx]
        
        # Define start and goal
        H, W = img_tensor.shape[1], img_tensor.shape[2]
        start = (int(H*0.1), int(W*0.1))
        goal = (int(H*0.9), int(W*0.9))
        
        paths = {}
        for bl in ['b1_euclidean', 'proposed']:
            path, data, fusion_dict = pipeline.run(img_tensor, start_coords=start, goal_coords=goal, run_baseline=bl)
            paths[bl] = path
            
        # Visualizations
        generate_visualizations(img_tensor, data, fusion_dict, paths, results_dir, prefix=f"ctx_{idx}")
        print(f"  CTX[{idx}] Processed. Plots saved.")
        
    print(f"Demo completed. Visualizations saved to {results_dir}")

if __name__ == "__main__":
    main()
