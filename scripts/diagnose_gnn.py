"""
diagnose_gnn.py — Diagnose the trained GATv2 checkpoint:
  1. Check whether the GNN meaningfully adjusts H_final (vs copying it).
  2. Sweep decision thresholds to find the optimal precision/recall/F1 point.

Task (v3): BCE binary classification — label = (H_final > 0.5) per node.

Run: python scripts/diagnose_gnn.py
"""
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
from torch_geometric.loader import DataLoader as PyGDataLoader
from src.utils.config import load_config
from src.utils.seed import set_seed, get_device
from src.data.loaders.precomputed_graph_dataset import PrecomputedGraphDataset
from src.models.gnn.gatv2 import PAGATv2

H_FINAL_IDX       = 7    # position of mean_H_final in 14-dim node feature vector
H_FINAL_THRESHOLD = 0.5  # label threshold used during training


def main():
    base_cfg = load_config("configs/base.yaml")
    gat_cfg  = load_config("configs/gnn/gatv2.yaml")
    set_seed(base_cfg.project.seed)
    device = get_device(base_cfg.project.device)

    # ── Load checkpoint ────────────────────────────────────────────────────────
    ckpt_path = Path(base_cfg.paths.checkpoints) / "gnn_fast" / "best_gat_model.pth"
    model = PAGATv2(gat_cfg).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Print all available checkpoint metrics (robust to any key set)
    metrics    = ckpt.get("metrics", {})
    metric_str = " | ".join(f"{k}={v:.4f}" for k, v in metrics.items()) or "no metrics stored"
    print(f"Loaded checkpoint from epoch {ckpt['epoch']} | {metric_str}")

    # ── Load val graphs ────────────────────────────────────────────────────────
    graphs_dir = Path(base_cfg.paths.processed) / "graphs"
    val_ds     = PrecomputedGraphDataset.from_split_dir(graphs_dir, "val")
    loader     = PyGDataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0)

    all_diffs      = []   # |GNN_output - H_final_input|
    all_h_final_in = []   # H_final from input feature 7
    all_gnn_out    = []   # GNN predicted risk score
    all_targets    = []   # binary labels (H_final > 0.5)

    n_batches = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            if data.x.size(0) == 0:
                continue

            preds      = model(data.x, data.edge_index, data.edge_attr)  # (N,)
            h_final_in = data.x[:, H_FINAL_IDX]                          # (N,)
            targets    = (h_final_in > H_FINAL_THRESHOLD).float()

            diff = (preds - h_final_in).abs()
            all_diffs.extend(diff.cpu().numpy().tolist())
            all_h_final_in.extend(h_final_in.cpu().numpy().tolist())
            all_gnn_out.extend(preds.cpu().numpy().tolist())
            all_targets.extend(targets.cpu().numpy().tolist())

            n_batches += 1
            if n_batches >= 10:   # 10 batches × 16 graphs is sufficient
                break

    diffs   = np.array(all_diffs)
    h_in    = np.array(all_h_final_in)
    gnn_out = np.array(all_gnn_out)
    targets = np.array(all_targets)

    # ── Section 1: Copy-detection ──────────────────────────────────────────────
    print("\n── GNN vs H_final Input Comparison ──────────────────────────")
    print(f"  Nodes analysed          : {len(diffs):,}")
    print(f"  Mean |GNN - H_final|    : {diffs.mean():.6f}  ← 0 means copying")
    print(f"  Max  |GNN - H_final|    : {diffs.max():.6f}")
    print(f"  Nodes with diff > 0.05  : {(diffs > 0.05).sum():,}  ({100*(diffs > 0.05).mean():.1f}%)")
    print(f"  Nodes with diff > 0.10  : {(diffs > 0.10).sum():,}  ({100*(diffs > 0.10).mean():.1f}%)")

    # ── Section 2: Input distribution ─────────────────────────────────────────
    print()
    print("── H_final Input Distribution ────────────────────────────────")
    print(f"  Mean H_final (input)    : {h_in.mean():.4f}")
    print(f"  Std  H_final (input)    : {h_in.std():.4f}")
    print(f"  Nodes H_final > 0.5     : {(h_in > 0.5).sum():,}  ({100*(h_in > 0.5).mean():.1f}%)  ← positive class")
    print(f"  Nodes H_final > 0.7     : {(h_in > 0.7).sum():,}  ({100*(h_in > 0.7).mean():.1f}%)")

    # ── Section 3: GNN output distribution ────────────────────────────────────
    print()
    print("── GNN Output Distribution ───────────────────────────────────")
    print(f"  Mean GNN output         : {gnn_out.mean():.4f}")
    print(f"  Std  GNN output         : {gnn_out.std():.4f}")
    print(f"  Nodes GNN > 0.5         : {(gnn_out > 0.5).sum():,}  ({100*(gnn_out > 0.5).mean():.1f}%)")
    print(f"  Nodes GNN > 0.7         : {(gnn_out > 0.7).sum():,}  ({100*(gnn_out > 0.7).mean():.1f}%)")

    # ── Section 4: Threshold sweep ────────────────────────────────────────────
    print()
    print("── Threshold Sweep (Precision / Recall / F1) ─────────────────")
    print(f"  {'Thresh':>7}  {'Recall':>7}  {'Precision':>9}  {'F1':>6}  {'Flagged%':>9}")
    print(f"  {'-'*7}  {'-'*7}  {'-'*9}  {'-'*6}  {'-'*9}")

    best_f1, best_thresh = 0.0, 0.5
    for t in np.arange(0.25, 0.76, 0.05):
        p_hard    = gnn_out > t
        tp        = float(( p_hard & (targets == 1)).sum())
        fp        = float(( p_hard & (targets == 0)).sum())
        fn        = float((~p_hard & (targets == 1)).sum())
        recall    = tp / (tp + fn + 1e-8)
        precision = tp / (tp + fp + 1e-8)
        f1        = 2 * precision * recall / (precision + recall + 1e-8)
        flagged   = 100 * p_hard.mean()
        marker    = " << best F1" if f1 > best_f1 else ""
        if f1 > best_f1:
            best_f1, best_thresh = f1, float(t)
        print(f"  {t:>7.2f}  {recall:>7.4f}  {precision:>9.4f}  {f1:>6.4f}  {flagged:>8.1f}%{marker}")

    # ── Section 5: Verdict ────────────────────────────────────────────────────
    mean_diff = diffs.mean()
    print()
    if mean_diff < 0.005:
        verdict = "[FAIL] COPYING  -- GNN output is nearly identical to H_final input. No neighbourhood learning."
    elif mean_diff < 0.02:
        verdict = "[WARN] MINIMAL  -- Small but real adjustment. GNN adds marginal spatial smoothing."
    else:
        verdict = "[OK]   LEARNING -- GNN meaningfully adjusts H_final using neighbourhood context."
    print(f"VERDICT: {verdict}")

    print()
    print("── Recommended Action ────────────────────────────────────────")
    if mean_diff < 0.02:
        print("  The GNN is not adding meaningful refinement beyond H_final.")
        print("  Root cause: H_final is in the input features (index 7).")
        print("  Fix: Remove H_final from input features before training,")
        print("       so the GNN MUST use neighbourhood context to reconstruct it.")
        print("  → Run: python scripts/retrain_without_h_final.py")
    else:
        print(f"  The GNN IS refining H_final using neighbourhood context.")
        print(f"  Best F1 threshold : {best_thresh:.2f}  (F1={best_f1:.4f})")
        print(f"  Consider setting inference threshold={best_thresh:.2f} in configs/gnn/gatv2.yaml")
        print(f"  Next step        : python src/evaluation/evaluate_ai4mars.py")


if __name__ == "__main__":
    main()
