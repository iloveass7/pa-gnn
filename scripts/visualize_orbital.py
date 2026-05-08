"""
visualize_orbital.py  —  Orbital Imagery GNN Node & Path Visualisation
=======================================================================
Shows, for each CTX (or HiRISE) orbital tile:

  [Left]   Original orbital image + SLIC superpixel boundaries
           Nodes coloured by GNN risk score (green=safe → red=hazardous)
           Deactivated (blocked) nodes filled solid red
  [Center] GNN risk heatmap reprojected to pixel space
  [Right]  Path comparison overlaid on the orbital image:
           - Red dashed  = B1 Euclidean (naive straight line, ignores terrain)
           - Cyan solid  = PA-GNN proposed (routes around hazardous nodes)

Usage:
    # CTX (default, 3 tiles)
    python scripts/visualize_orbital.py

    # CTX, more tiles
    python scripts/visualize_orbital.py --n_samples 6 --dataset ctx

    # HiRISE
    python scripts/visualize_orbital.py --n_samples 5 --dataset hirise

Output:
    results/orbital_vis/orbital_{n:03d}.png
"""

import sys, argparse
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from skimage.segmentation import mark_boundaries

from src.utils.config import load_config
from src.utils.seed import set_seed, get_device
from src.inference.pipeline import PA_GNN_Pipeline

H_FINAL_IDX = 7   # index of mean_H_final in the 14-dim node feature vector


# ── helpers ──────────────────────────────────────────────────────────────────

def risk_to_pixel_map(label_map, node_values):
    """Back-project per-node scalars onto pixel space via the label_map."""
    out = np.zeros(label_map.shape, dtype=np.float32)
    for node_id, val in enumerate(node_values):
        out[label_map == (node_id + 1)] = val
    return out


def run_both_paths(pipeline, img_tensor, device):
    """Return (gnn_scores, data, fusion_dict, path_b1, path_proposed)."""
    # Proposed path (with GATv2 deactivation)
    H, W = img_tensor.shape[1], img_tensor.shape[2]
    start = (int(H * 0.1), int(W * 0.1))
    goal  = (int(H * 0.9), int(W * 0.9))

    path_proposed, data, fusion_dict, _ = pipeline.run(
        img_tensor, start_coords=start, goal_coords=goal, run_baseline='proposed'
    )
    path_b1, _, _, _ = pipeline.run(
        img_tensor, start_coords=start, goal_coords=goal, run_baseline='b1_euclidean'
    )

    # Re-run GATv2 to get raw node scores
    with torch.no_grad():
        data_dev = data.to(device)
        gnn_preds = pipeline.gat_model(data_dev.x, data_dev.edge_index, data_dev.edge_attr)
    gnn_scores = gnn_preds.cpu().numpy()

    return gnn_scores, data, fusion_dict, path_b1, path_proposed


def make_figure(img_np, data, fusion_dict, gnn_scores, path_b1, path_proposed,
                deact_thresh, sample_idx, source_label):
    """Create and return a 3-panel figure."""
    label_map  = data.label_map.cpu().numpy()   # (H, W) 1-indexed
    pos        = data.pos.cpu().numpy()          # (N, 2) centroid (y, x)
    active     = data.active_mask.cpu().numpy()  # (N,) bool

    N = gnn_scores.shape[0]
    H, W = img_np.shape

    # ── GNN risk pixel map ────────────────────────────────────────────────────
    gnn_pixel = risk_to_pixel_map(label_map, gnn_scores)

    # ── Colour palette for nodes ──────────────────────────────────────────────
    cmap_risk  = plt.cm.RdYlGn_r
    norm_risk  = Normalize(vmin=0, vmax=1)

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    fig.suptitle(
        f"{source_label} Sample {sample_idx:03d}  ·  PA-GNN Orbital Visualisation",
        fontsize=13, fontweight='bold', y=1.01
    )

    # ──────────────────────────────────────────────────────────────────────────
    # Panel 0: Superpixel graph with GNN risk-coloured nodes
    # ──────────────────────────────────────────────────────────────────────────
    ax = axes[0]
    ax.set_title("Superpixel Graph  ·  Node Risk (GATv2)", fontsize=11)

    # base image with boundaries
    bound_img = mark_boundaries(img_np, label_map, color=(0.3, 0.3, 0.3),
                                mode='inner', background_label=0)
    ax.imshow(bound_img, cmap='gray' if bound_img.ndim == 2 else None)

    # fill each superpixel with its risk colour, transparent overlay
    overlay = np.zeros((H, W, 4), dtype=np.float32)
    for nid in range(N):
        mask = label_map == (nid + 1)
        r, g, b, a_ = cmap_risk(norm_risk(gnn_scores[nid]))
        blocked = gnn_scores[nid] > deact_thresh
        alpha = 0.65 if blocked else 0.35
        overlay[mask] = [r, g, b, alpha]

    ax.imshow(overlay, interpolation='nearest')

    # draw node centroids: green circle = active, red X = deactivated
    for nid in range(N):
        cy, cx = pos[nid]
        blocked = gnn_scores[nid] > deact_thresh
        if blocked:
            ax.plot(cx, cy, 'rx', markersize=5, markeredgewidth=1.2, alpha=0.85)
        else:
            ax.plot(cx, cy, 'o', color='lime', markersize=3, alpha=0.5)

    # colorbar
    sm = ScalarMappable(cmap=cmap_risk, norm=norm_risk)
    sm.set_array([])
    cb = plt.colorbar(sm, ax=ax, fraction=0.04, pad=0.02, shrink=0.85)
    cb.set_label('GNN Risk Score', fontsize=9)

    # legend
    patch_safe = mpatches.Patch(facecolor='lime',   label='Active node (risk ≤ 0.70)')
    patch_haz  = mpatches.Patch(facecolor='red',    label='Deactivated (risk > 0.70)')
    ax.legend(handles=[patch_safe, patch_haz], fontsize=7, loc='upper left',
              framealpha=0.75)
    ax.axis('off')

    # ──────────────────────────────────────────────────────────────────────────
    # Panel 1: GNN risk heatmap (pixel space)
    # ──────────────────────────────────────────────────────────────────────────
    ax = axes[1]
    ax.set_title("GNN Risk Heatmap  (pixel-reprojected)", fontsize=11)
    im = ax.imshow(gnn_pixel, cmap='RdYlGn_r', vmin=0, vmax=1, interpolation='nearest')
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02, shrink=0.85, label='Risk')
    # contour at deactivation threshold
    ax.contour(gnn_pixel, levels=[deact_thresh],
               colors=['white'], linewidths=1.2, linestyles='--', alpha=0.8)
    ax.axis('off')

    # ──────────────────────────────────────────────────────────────────────────
    # Panel 2: Path comparison
    # ──────────────────────────────────────────────────────────────────────────
    ax = axes[2]
    ax.set_title("Path Comparison  ·  B1 Euclidean vs PA-GNN", fontsize=11)
    ax.imshow(img_np, cmap='gray')

    # lightly tint deactivated nodes red
    red_overlay = np.zeros((H, W, 4), dtype=np.float32)
    for nid in range(N):
        if gnn_scores[nid] > deact_thresh:
            red_overlay[label_map == (nid + 1)] = [1, 0, 0, 0.35]
    ax.imshow(red_overlay, interpolation='nearest')

    def draw_path(path_details, color, label, style='-', lw=2.2, zorder=5):
        if not path_details:
            return
        xs = [p['pos'][1] for p in path_details]
        ys = [p['pos'][0] for p in path_details]
        ax.plot(xs, ys, color=color, linewidth=lw, linestyle=style,
                label=label, zorder=zorder, alpha=0.9)
        ax.plot(xs[0],  ys[0],  's', color=color, markersize=9, zorder=zorder+1)
        ax.plot(xs[-1], ys[-1], '^', color=color, markersize=9, zorder=zorder+1)

    draw_path(path_b1,       color='tomato',     label='B1 Euclidean (naive)',
              style='--', lw=2.0, zorder=5)
    draw_path(path_proposed, color='deepskyblue', label='PA-GNN (risk-aware)',
              style='-',  lw=2.5, zorder=6)

    if path_b1 is None:
        ax.text(0.5, 0.55, "B1: NO PATH", color='tomato', fontsize=9,
                ha='center', transform=ax.transAxes)
    if path_proposed is None:
        ax.text(0.5, 0.45, "PA-GNN: NO PATH\n(all routes blocked)", color='deepskyblue',
                fontsize=9, ha='center', transform=ax.transAxes)

    ax.legend(fontsize=8, loc='upper left', framealpha=0.8)
    ax.axis('off')

    plt.tight_layout()
    return fig


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_samples', type=int,   default=3)
    parser.add_argument('--dataset',   type=str,   default='ctx',
                        choices=['ctx', 'hirise'])
    parser.add_argument('--out_dir',   type=str,   default='results/orbital_vis')
    args = parser.parse_args()

    base_cfg   = load_config('configs/base.yaml')
    cnn_cfg    = load_config('configs/cnn/mobilenetv3.yaml')
    phys_cfg   = load_config('configs/physics.yaml')
    fusion_cfg = load_config('configs/fusion/adaptive_fusion.yaml')
    gat_cfg    = load_config('configs/gnn/gatv2.yaml')

    device = get_device(base_cfg.project.device)
    set_seed(base_cfg.project.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fusion_ckpt = Path(base_cfg.paths.checkpoints) / 'fusion'   / 'best_model.pth'
    gat_ckpt    = Path(base_cfg.paths.checkpoints) / 'gnn_fast' / 'best_gat_model.pth'
    print(f"Fusion ckpt : {'FOUND' if fusion_ckpt.exists() else 'NOT FOUND'}")
    print(f"GATv2 ckpt  : {'FOUND' if gat_ckpt.exists() else 'NOT FOUND'}")

    pipeline = PA_GNN_Pipeline(
        base_cfg, cnn_cfg, phys_cfg, fusion_cfg, gat_cfg, device,
        fusion_ckpt=str(fusion_ckpt) if fusion_ckpt.exists() else None,
        gat_ckpt=str(gat_ckpt)    if gat_ckpt.exists()    else None,
    )

    deact_thresh = 1.0 - gat_cfg.graph.deactivation_threshold   # 0.70

    # ── Load dataset ─────────────────────────────────────────────────────────
    if args.dataset == 'ctx':
        from src.data.loaders.ctx_loader import CTXDataset
        ctx_cfg = load_config('configs/datasets/ctx.yaml')
        ds      = CTXDataset.from_config(base_cfg, ctx_cfg, max_tiles=200)
        indices = ds.select_demo_tiles(n=args.n_samples, seed=base_cfg.project.seed)
        source_label = 'CTX Orbital'
        def get_img(i):
            img_tensor, _ = ds[i]
            return img_tensor
    else:
        from src.data.loaders.hirise_loader import HiRISEDataset
        hirise_cfg = load_config('configs/datasets/hirise.yaml')
        ds         = HiRISEDataset.from_config(base_cfg, hirise_cfg)
        # pick spread across dataset
        step    = max(1, len(ds) // args.n_samples)
        indices = list(range(0, min(args.n_samples * step, len(ds)), step))[:args.n_samples]
        source_label = 'HiRISE Orbital'
        def get_img(i):
            img_tensor, _, _ = ds[i]
            return img_tensor

    print(f"\nGenerating {len(indices)} visualisations "
          f"({source_label}) → {out_dir}/\n")

    for rank, idx in enumerate(indices):
        img_tensor = get_img(idx)

        # grayscale for display
        if img_tensor.shape[0] == 3:
            img_np = img_tensor.mean(dim=0).cpu().numpy()
        else:
            img_np = img_tensor.squeeze().cpu().numpy()

        try:
            gnn_scores, data, fusion_dict, path_b1, path_proposed = \
                run_both_paths(pipeline, img_tensor, device)
        except Exception as e:
            print(f"  [WARN] sample idx={idx}: {e}")
            import traceback; traceback.print_exc()
            continue

        n_blocked  = int((gnn_scores > deact_thresh).sum())
        n_total    = len(gnn_scores)
        sr_b1      = "FOUND" if path_b1       else "NO PATH"
        sr_prop    = "FOUND" if path_proposed else "NO PATH"
        print(f"  [{rank+1}/{len(indices)}] idx={idx:4d} | "
              f"nodes={n_total} | blocked={n_blocked} ({100*n_blocked/n_total:.1f}%) | "
              f"B1={sr_b1} | PA-GNN={sr_prop}")

        fig = make_figure(img_np, data, fusion_dict, gnn_scores,
                          path_b1, path_proposed,
                          deact_thresh, rank, source_label)

        save_path = out_dir / f"orbital_{rank:03d}_{args.dataset}.png"
        fig.savefig(save_path, dpi=140, bbox_inches='tight')
        plt.close(fig)
        print(f"    Saved: {save_path}")

    print(f"\nDone. Open {out_dir}/ to review the figures.")


if __name__ == '__main__':
    main()
