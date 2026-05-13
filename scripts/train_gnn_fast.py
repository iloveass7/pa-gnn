"""
train_gnn_fast.py — Fast GNN training using pre-computed graphs.

GNN Task (v3): Binary BCE node classification with H_final-derived labels.
─────────────────────────────────────────────────────────────────────────
CURRENT APPROACH (v3):
  Target  : (data.x[:, 7] > 0.5).float()  ← H_final-derived binary labels
  Positive: 15.5% of nodes (5.5:1 imbalance) → positive_weight=5.0
  Loss    : weighted BCE (NOT SmoothL1 regression — that was v2 and collapsed)
  Result  : GNN refines H_final using 2-hop neighbourhood attention;
            output spans full [0,1] range; A* deactivates nodes where risk > 0.70

PREVIOUS APPROACHES (abandoned):
  v1 — Binary BCE on raw AI4Mars labels → 1.9% positive, 47:1 imbalance → oscillation
  v2 — SmoothL1 regression on H_final  → output collapsed to mean (std=0.054)
──────────────────────────────────────────────────────────────────────────
"""


import sys
import argparse
import json
import time
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.metrics import roc_auc_score
from torch_geometric.loader import DataLoader as PyGDataLoader

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import load_config
from src.utils.seed import set_seed, get_device
from src.utils.logger import get_logger, setup_logger
from src.data.loaders.precomputed_graph_dataset import PrecomputedGraphDataset
from src.models.gnn.gatv2 import PAGATv2

# H_FINAL_FEAT_IDX: position of mean_H_final in the 14-dim node feature vector
# See node_features.py: features[i, 7] = h_f_np[mask].mean()
H_FINAL_FEAT_IDX = 7
H_FINAL_THRESHOLD = 0.5


@torch.no_grad()
def evaluate_gnn_fast(model, loader, device, positive_weight):
    model.eval()
    all_targets = []
    all_preds   = []
    total_loss  = 0.0
    n_batches   = 0

    for data in loader:
        data = data.to(device)
        if data.x.size(0) == 0:
            continue

        preds   = model(data.x, data.edge_index, data.edge_attr)  # (N,)
        targets = (data.x[:, H_FINAL_FEAT_IDX] > H_FINAL_THRESHOLD).float()

        weight = torch.ones_like(targets)
        weight[targets > 0.5] = positive_weight
        loss = F.binary_cross_entropy(preds, targets, weight=weight)

        total_loss += loss.item()
        n_batches  += 1
        all_targets.extend(targets.cpu().numpy().tolist())
        all_preds.extend(preds.cpu().numpy().tolist())

    avg_loss = total_loss / max(n_batches, 1)
    t = np.array(all_targets)
    p = np.array(all_preds)

    metrics = {"val_loss": avg_loss}
    if len(t) > 1:
        try:
            metrics["val_auc_roc"] = roc_auc_score(t, p)
        except ValueError:
            metrics["val_auc_roc"] = 0.5

        p_hard = p > 0.5
        tp = ((p_hard) & (t == 1)).sum()
        fn = ((~p_hard) & (t == 1)).sum()
        fp = ((p_hard) & (t == 0)).sum()
        metrics["val_hazard_recall"]    = float(tp / (tp + fn + 1e-8))
        metrics["val_hazard_precision"] = float(tp / (tp + fp + 1e-8))
        metrics["pct_flagged"]          = float(p_hard.mean() * 100)

    return metrics


def train_epoch(model, loader, optimizer, device, epoch, logger, positive_weight):
    model.train()
    total_loss = 0.0
    n_batches  = 0
    t0         = time.time()

    for batch_idx, data in enumerate(loader):
        data = data.to(device)
        if data.x.size(0) == 0:
            continue

        # Binary target derived from H_final (feature 7) — no recomputation needed
        # 15.5% positive rate → 5.5:1 imbalance → managed with positive_weight
        targets = (data.x[:, H_FINAL_FEAT_IDX] > H_FINAL_THRESHOLD).float()

        optimizer.zero_grad()
        preds = model(data.x, data.edge_index, data.edge_attr)

        weight = torch.ones_like(targets)
        weight[targets > 0.5] = positive_weight
        loss = F.binary_cross_entropy(preds, targets, weight=weight)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches  += 1

        if batch_idx % 50 == 0:
            logger.info(f"Epoch {epoch} [{batch_idx}/{len(loader)}] Loss: {loss.item():.4f}")

    avg_loss   = total_loss / max(n_batches, 1)
    elapsed    = time.time() - t0
    logger.info(f"Epoch {epoch} | Train Loss: {avg_loss:.4f} | {elapsed:.1f}s | {n_batches/elapsed:.1f} batches/s")
    return avg_loss


def main():
    parser = argparse.ArgumentParser(
        description="Fast GNN training — SmoothL1 regression on H_final per node"
    )
    parser.add_argument("--graphs_dir",  type=str, default=None,
                        help="Root dir of precomputed graphs (default: from base_cfg)")
    parser.add_argument("--base_cfg",    type=str, default="configs/base.yaml")
    parser.add_argument("--gat_cfg",     type=str, default="configs/gnn/gatv2.yaml")
    parser.add_argument("--batch_size",  type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--gat_ckpt",    type=str, default=None,
                        help="Resume from checkpoint (.pth)")
    args = parser.parse_args()

    base_cfg = load_config(args.base_cfg)
    gat_cfg  = load_config(args.gat_cfg)

    set_seed(base_cfg.project.seed)
    device = get_device(base_cfg.project.device)

    checkpoint_dir = Path(base_cfg.paths.checkpoints) / "gnn_fast"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(base_cfg.paths.logs) / "gnn_fast"
    log_dir.mkdir(parents=True, exist_ok=True)

    setup_logger(log_file=log_dir / "train_gnn_fast.log")
    logger = get_logger("TrainGNN_Fast")
    pos_w = gat_cfg.training.loss.positive_weight
    logger.info(f"Starting Fast GNN Training (H_final binary classification) on {device}")
    logger.info(
        f"Task: BCE on (H_final > {H_FINAL_THRESHOLD}) per node\n"
        f"      Positive rate ~15.5% (5.5:1 imbalance), positive_weight={pos_w}"
    )

    graphs_dir = args.graphs_dir or str(Path(base_cfg.paths.processed) / "graphs")
    logger.info(f"Loading precomputed graphs from: {graphs_dir}")
    train_ds = PrecomputedGraphDataset.from_split_dir(graphs_dir, "train")
    val_ds   = PrecomputedGraphDataset.from_split_dir(graphs_dir, "val")
    logger.info(f"Train: {len(train_ds)} graphs | Val: {len(val_ds)} graphs")

    train_loader = PyGDataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )
    val_loader = PyGDataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
    )

    gat_model = PAGATv2(gat_cfg).to(device)
    logger.info(
        f"Model: PAGATv2 | "
        f"Params: {sum(p.numel() for p in gat_model.parameters()):,}"
    )

    optimizer = optim.Adam(
        gat_model.parameters(),
        lr=gat_cfg.training.learning_rate,
        weight_decay=gat_cfg.training.weight_decay,
    )

    # Scheduler: reduce LR when val_hazard_recall stops improving (higher = better)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-6
    )

    start_epoch      = 1
    best_recall      = 0.0
    patience_counter = 0
    history          = defaultdict(list)

    if args.gat_ckpt and Path(args.gat_ckpt).exists():
        ckpt = torch.load(args.gat_ckpt, map_location=device, weights_only=False)
        gat_model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch  = ckpt.get("epoch", 0) + 1
        best_recall  = ckpt.get("metrics", {}).get("val_hazard_recall", 0.0)
        logger.info(
            f"Resumed from {args.gat_ckpt} "
            f"(epoch {start_epoch}, val_hazard_recall {best_recall:.4f})"
        )

    epochs   = gat_cfg.training.epochs
    patience = gat_cfg.training.early_stopping.patience
    logger.info(f"Training for up to {epochs} epochs (patience={patience})")

    for epoch in range(start_epoch, epochs + 1):

        train_loss = train_epoch(
            gat_model, train_loader, optimizer, device, epoch, logger, pos_w
        )
        history["train_loss"].append(train_loss)

        val_metrics = evaluate_gnn_fast(gat_model, val_loader, device, pos_w)
        for k, v in val_metrics.items():
            history[k].append(v)

        logger.info(
            f"Epoch {epoch} Val: " +
            ", ".join(f"{k}: {v:.4f}" for k, v in val_metrics.items())
        )

        # Step scheduler on val_hazard_recall (higher = better)
        current_recall = val_metrics.get("val_hazard_recall", 0.0)
        scheduler.step(current_recall)

        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(f"  LR: {current_lr:.2e}")

        # Checkpoint & early stopping — monitor val_hazard_recall (higher = better)
        if current_recall > best_recall:
            best_recall      = current_recall
            patience_counter = 0
            torch.save({
                "epoch":                epoch,
                "model_state_dict":     gat_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "metrics":              val_metrics,
            }, checkpoint_dir / "best_gat_model.pth")
            logger.info(f"  ✓ New best saved (val_hazard_recall={best_recall:.4f})")
        else:
            patience_counter += 1
            logger.info(f"  No improvement ({patience_counter}/{patience})")
            if patience_counter >= patience:
                logger.info("Early stopping triggered.")
                break

        with open(checkpoint_dir / "history.json", "w") as f:
            json.dump(dict(history), f, indent=2)

    logger.info(f"Training complete. Best val_hazard_recall: {best_recall:.4f}")
    logger.info(f"Checkpoint: {checkpoint_dir / 'best_gat_model.pth'}")


if __name__ == "__main__":
    main()