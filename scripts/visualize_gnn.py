"""
visualize_gnn.py  -  GNN Risk Map Visualisation
=================================================
Generates a 6-panel figure per sample showing the full pipeline:
  [0] Original image
  [1] H_physics   (physics-only risk)
  [2] H_learned   (CNN risk heatmap)
  [3] H_final     (fused risk — GNN input)
  [4] GNN output  (GATv2 refined risk per superpixel, reprojected to pixels)
  [5] GNN Delta   (H_final_reprojected - GNN_output, shows where GNN changed things)

Usage:
    python scripts/visualize_gnn.py [--n_samples 5] [--out_dir results/gnn_vis]

Saved files (per sample):
    results/gnn_vis/sample_{i:03d}.png
"""

import sys
import argparse
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from src.utils.config import load_config
from src.utils.seed import set_seed, get_device
from src.data.loaders.ai4mars_loader import AI4MarsDataset
from src.inference.pipeline import PA_GNN_Pipeline

H_FINAL_IDX = 7   # index of mean_H_final in the 14-dim node feature vector


def risk_to_pixel_map(label_map, node_values, n_nodes):
    """Reproject per-node scalar values back onto pixels using the label_map."""
    h, w = label_map.shape
    out = np.zeros((h, w), dtype=np.float32)
    for node_id in range(n_nodes):
        # label_map is 1-indexed (start_label=1); node i corresponds to label i+1
        mask = label_map == (node_id + 1)
        out[mask] = node_values[node_id]
    return out


def visualize_sample(pipeline, img_tensor, sample_idx, out_path, device):
    """Run pipeline on one image and save the 6-panel figure."""
    H, W = img_tensor.shape[1], img_tensor.shape[2]
    start = (int(H * 0.1), int(W * 0.1))
    goal  = (int(H * 0.9), int(W * 0.9))

    # Run full pipeline (proposed baseline to get GNN output)
    path_details, data, fusion_dict, _ = pipeline.run(
        img_tensor, start_coords=start, goal_coords=goal, run_baseline='proposed'
    )

    # --- Original image (grayscale) ---
    if img_tensor.shape[0] == 3:
        img_np = img_tensor.mean(dim=0).cpu().numpy()
    else:
        img_np = img_tensor.squeeze().cpu().numpy()

    # --- Physics / CNN / Fusion maps ---
    h_phys  = fusion_dict['h_physics'].squeeze().cpu().numpy()
    h_learn = fusion_dict['h_learned'].squeeze().cpu().numpy()
    h_final = fusion_dict['h_final'].squeeze().cpu().numpy()

    # --- GNN output reprojected to pixel space ---
    label_map = data.label_map.cpu().numpy()           # (H, W) 1-indexed
    n_nodes   = data.x.shape[0]

    # GNN output: re-run model on the (already-moved-to-device) data
    with torch.no_grad():
        data_dev = data.to(device)
        gnn_preds = pipeline.gat_model(data_dev.x, data_dev.edge_index, data_dev.edge_attr)
    gnn_scores = gnn_preds.cpu().numpy()               # (N,)
    h_final_nodes = data.x[:, H_FINAL_IDX].cpu().numpy()  # (N,)

    gnn_pixel   = risk_to_pixel_map(label_map, gnn_scores, n_nodes)
    delta_pixel = gnn_pixel - risk_to_pixel_map(label_map, h_final_nodes, n_nodes)

    # --- Plot ---
    fig = plt.figure(figsize=(22, 8))
    fig.suptitle(f"Sample {sample_idx:03d}  |  GNN Pipeline Visualisation", fontsize=14, fontweight='bold')
    gs = gridspec.GridSpec(1, 6, figure=fig, wspace=0.04)

    panels = [
        (img_np,    "Original",          'gray',      None,   None),
        (h_phys,    "H_physics",         'hot',       0.0,    1.0),
        (h_learn,   "H_learned (CNN)",   'hot',       0.0,    1.0),
        (h_final,   "H_final (Fused)",   'hot',       0.0,    1.0),
        (gnn_pixel, "GNN Output (GATv2)",'RdYlGn_r', 0.0,    1.0),
        (delta_pixel,"GNN Delta\n(GNN - H_final)",'RdBu_r',-0.5, 0.5),
    ]

    for col, (img, title, cmap, vmin, vmax) in enumerate(panels):
        ax = fig.add_subplot(gs[0, col])
        im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')
        ax.set_title(title, fontsize=10, pad=4)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02, shrink=0.85)

    # Overlay path on GNN panel (panel 4)
    ax_gnn = fig.axes[4]   # 0-indexed axes after gridspec
    if path_details is not None and len(path_details) > 0:
        xs = [p['pos'][1] for p in path_details]  # col
        ys = [p['pos'][0] for p in path_details]  # row
        ax_gnn.plot(xs, ys, 'w--', linewidth=1.5, alpha=0.8)
        ax_gnn.plot(xs[0],  ys[0],  'bs', markersize=7, label='Start')
        ax_gnn.plot(xs[-1], ys[-1], 'g^', markersize=7, label='Goal')
        ax_gnn.legend(fontsize=7, loc='upper left')
    else:
        ax_gnn.text(0.5, 0.5, "NO PATH", color='red', fontsize=9,
                    ha='center', va='center', transform=ax_gnn.transAxes,
                    bbox=dict(facecolor='white', alpha=0.6))

    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualise GNN risk refinement on AI4Mars test samples")
    parser.add_argument("--n_samples", type=int, default=5,  help="Number of samples to visualise (default: 5)")
    parser.add_argument("--out_dir",   type=str, default="results/gnn_vis", help="Output directory")
    parser.add_argument("--split",     type=str, default="test", choices=["train", "val", "test"])
    args = parser.parse_args()

    base_cfg    = load_config("configs/base.yaml")
    cnn_cfg     = load_config("configs/cnn/mobilenetv3.yaml")
    phys_cfg    = load_config("configs/physics.yaml")
    fusion_cfg  = load_config("configs/fusion/adaptive_fusion.yaml")
    gat_cfg     = load_config("configs/gnn/gatv2.yaml")
    ai4mars_cfg = load_config("configs/datasets/ai4mars.yaml")

    device = get_device(base_cfg.project.device)
    set_seed(base_cfg.project.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load pipeline with trained checkpoints
    fusion_ckpt = Path(base_cfg.paths.checkpoints) / "fusion"  / "best_model.pth"
    gat_ckpt    = Path(base_cfg.paths.checkpoints) / "gnn_fast" / "best_gat_model.pth"

    print(f"Fusion ckpt : {'FOUND' if fusion_ckpt.exists() else 'NOT FOUND'} ({fusion_ckpt})")
    print(f"GATv2 ckpt  : {'FOUND' if gat_ckpt.exists() else 'NOT FOUND'} ({gat_ckpt})")

    pipeline = PA_GNN_Pipeline(
        base_cfg, cnn_cfg, phys_cfg, fusion_cfg, gat_cfg, device,
        fusion_ckpt=str(fusion_ckpt) if fusion_ckpt.exists() else None,
        gat_ckpt=str(gat_ckpt)    if gat_ckpt.exists()    else None,
    )

    dataset = AI4MarsDataset.from_config(base_cfg, ai4mars_cfg, split=args.split)
    n = min(args.n_samples, len(dataset))
    print(f"\nGenerating {n} visualisations from '{args.split}' split -> {out_dir}/\n")

    for i in range(n):
        img_tensor, _, _ = dataset[i]
        out_path = out_dir / f"sample_{i:03d}.png"
        try:
            visualize_sample(pipeline, img_tensor, i, out_path, device)
        except Exception as e:
            print(f"  [WARN] Sample {i} failed: {e}")

    print(f"\nDone. Open {out_dir}/ to review.")


if __name__ == "__main__":
    main()
