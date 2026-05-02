"""
train_gnn_fast.py — Fast GNN training using pre-computed graphs.

Replaces the slow on-the-fly graph building in train_gnn.py.
Requires precompute_graphs.py to have been run first.

Speed difference:
    train_gnn.py      (on-the-fly):  ~0.8s per image → 8 days for 100 epochs
    train_gnn_fast.py (precomputed): ~0.01s per image → ~3-4 hours for 100 epochs

Key differences from train_gnn.py:
  - Uses PrecomputedGraphDataset instead of AI4MarsDataset
  - Uses PyG DataLoader instead of torch DataLoader (native graph batching)
  - No fusion model needed at training time (graphs already built)
  - Supports real batch sizes (default 32) instead of forced batch_size=1
  - Everything else (GATv2, weak labeling, loss, early stopping) is identical

Usage:
    # Step 1 — Run fusion training (if not done already):
    python scripts/train_fusion.py --cnn_ckpt checkpoints/cnn/best_model.pth

    # Step 2 — Precompute graphs (one time, ~3 hrs):
    python scripts/precompute_graphs.py --fusion_ckpt checkpoints/fusion/best_model.pth

    # Step 3 — Fast GNN training:
    python scripts/train_gnn_fast.py

    # Optional overrides:
    python scripts/train_gnn_fast.py \\
        --graphs_dir d:/Mars/pa-gnn/data/processed/graphs \\
        --batch_size 64 \\
        --gat_cfg configs/gnn/gatv2.yaml
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

# Add project root
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import load_config
from src.utils.seed import set_seed, get_device
from src.utils.logger import get_logger, setup_logger
from src.data.loaders.precomputed_graph_dataset import PrecomputedGraphDataset
from src.models.gnn.gatv2 import PAGATv2
from src.training.weak_labels import compute_weak_labels


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_gnn_fast(model, loader, device, gat_cfg):
    """
    Evaluate GNN on a precomputed graph dataset.
    Mathematically identical to evaluate_gnn() in train_gnn.py.
    """
    model.eval()

    all_targets = []
    all_preds   = []
    total_loss  = 0.0
    n_batches   = 0

    with torch.no_grad():
        for data in loader:
            data = data.to(device)

            if data.x.size(0) == 0:
                continue

            # Guard: y must be a valid tensor (dataset always provides one, but be safe)
            if data.y is None:
                continue

            # Forward
            preds = model(data.x, data.edge_index, data.edge_attr)

            # Active + valid mask
            valid_mask = (data.y >= 0) & data.active_mask
            if valid_mask.sum() == 0:
                continue

            valid_preds   = preds[valid_mask]
            valid_targets = data.y[valid_mask]
            hard_targets  = (valid_targets > 0.5).float()

            # Loss
            weight = torch.ones_like(hard_targets)
            weight[hard_targets > 0.5] = gat_cfg.training.loss.positive_weight
            loss = F.binary_cross_entropy(valid_preds, hard_targets, weight=weight)

            total_loss += loss.item()
            n_batches  += 1

            all_targets.extend(hard_targets.cpu().numpy().tolist())
            all_preds.extend(valid_preds.cpu().numpy().tolist())

    avg_loss = total_loss / max(n_batches, 1)
    metrics  = {"val_loss": avg_loss}

    if len(all_targets) > 1:
        t = np.array(all_targets)
        p = np.array(all_preds)

        try:
            metrics["val_auc_roc"] = roc_auc_score(t, p)
        except ValueError:
            metrics["val_auc_roc"] = 0.5

        p_hard = p > 0.5
        tp = ((p_hard == 1) & (t == 1)).sum()
        fn = ((p_hard == 0) & (t == 1)).sum()
        metrics["val_hazard_recall"] = float(tp / (tp + fn + 1e-8))

    return metrics


# ── Training ──────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, device, gat_cfg, epoch, logger):
    """One training epoch over precomputed graphs."""
    model.train()

    total_loss = 0.0
    n_batches  = 0
    t0         = time.time()

    for batch_idx, data in enumerate(loader):
        data = data.to(device)

        if data.x.size(0) == 0:
            continue

        node_labels = data.y
        active_mask = data.active_mask

        # Guard: skip if no ground truth in this batch
        if node_labels is None:
            continue

        # ── Weak labeling (same logic as train_gnn.py) ───────────────────
        if gat_cfg.training.weak_labeling.enabled:
            hard_labels = (node_labels > 0.5).float()
            hard_labels[node_labels < 0] = -1.0
            updated_labels = compute_weak_labels(
                data.edge_index,
                hard_labels,
                active_mask,
                hops=gat_cfg.training.weak_labeling.hops,
                weak_value=gat_cfg.training.weak_labeling.label_value
            )
        else:
            updated_labels = (node_labels > 0.5).float()
            updated_labels[node_labels < 0] = -1.0

        # ── Forward & loss ────────────────────────────────────────────────
        optimizer.zero_grad()
        preds = model(data.x, data.edge_index, data.edge_attr)

        valid_mask = (updated_labels >= 0) & active_mask
        if valid_mask.sum() == 0:
            continue

        valid_preds   = preds[valid_mask]
        valid_targets = updated_labels[valid_mask]

        weight = torch.ones_like(valid_targets)
        weight[valid_targets >= 0.9] = gat_cfg.training.loss.positive_weight   # confirmed hazard
        weight[(valid_targets > 0.5) & (valid_targets < 0.9)] = 1.0            # weak hazard

        loss = F.binary_cross_entropy(valid_preds, valid_targets, weight=weight)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches  += 1

        if batch_idx % 50 == 0:
            logger.info(f"Epoch {epoch} [{batch_idx}/{len(loader)}] Loss: {loss.item():.4f}")

    avg_loss   = total_loss / max(n_batches, 1)
    elapsed    = time.time() - t0
    # PyG DataLoader.batch_size can be None — count graphs processed instead
    throughput = n_batches / elapsed if elapsed > 0 else 0

    logger.info(f"Epoch {epoch} | Train Loss: {avg_loss:.4f} | "
                f"{elapsed:.1f}s | {throughput:.1f} batches/s")

    return avg_loss


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fast GNN training on precomputed graphs")
    parser.add_argument("--graphs_dir", type=str, default=None,
                        help="Root dir with train/val/test subdirs (default: <processed>/graphs)")
    parser.add_argument("--base_cfg",   type=str, default="configs/base.yaml")
    parser.add_argument("--gat_cfg",    type=str, default="configs/gnn/gatv2.yaml")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Graph batch size for PyG DataLoader (default: 32)")
    parser.add_argument("--num_workers",type=int, default=4)
    parser.add_argument("--gat_ckpt",   type=str, default=None,
                        help="Resume from existing GNN checkpoint (.pth)")
    args = parser.parse_args()

    base_cfg = load_config(args.base_cfg)
    gat_cfg  = load_config(args.gat_cfg)

    set_seed(base_cfg.project.seed)
    device = get_device(base_cfg.project.device)

    # ── Logging ──────────────────────────────────────────────────────────
    checkpoint_dir = Path(base_cfg.paths.checkpoints) / "gnn_fast"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(base_cfg.paths.logs) / "gnn_fast"
    log_dir.mkdir(parents=True, exist_ok=True)

    setup_logger(log_file=log_dir / "train_gnn_fast.log")
    logger = get_logger("TrainGNN_Fast")
    logger.info(f"Starting Fast GNN Training on {device}")

    # ── Dataset ──────────────────────────────────────────────────────────
    graphs_dir = args.graphs_dir or str(Path(base_cfg.paths.processed) / "graphs")

    logger.info(f"Loading precomputed graphs from: {graphs_dir}")
    train_ds = PrecomputedGraphDataset.from_split_dir(graphs_dir, "train")
    val_ds   = PrecomputedGraphDataset.from_split_dir(graphs_dir, "val")
    logger.info(f"Train: {len(train_ds)} graphs | Val: {len(val_ds)} graphs")

    # PyG DataLoader — handles variable-size graph batching natively
    train_loader = PyGDataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = PyGDataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # ── Model & optimizer ────────────────────────────────────────────────
    gat_model = PAGATv2(gat_cfg).to(device)

    optimizer = optim.Adam(
        gat_model.parameters(),
        lr=gat_cfg.training.learning_rate,
        weight_decay=gat_cfg.training.weight_decay,
    )

    start_epoch = 1
    best_auc    = 0.0

    if args.gat_ckpt and Path(args.gat_ckpt).exists():
        ckpt = torch.load(args.gat_ckpt, map_location=device, weights_only=True)
        gat_model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_auc    = ckpt.get("metrics", {}).get("val_auc_roc", 0.0)
        logger.info(f"Resumed from {args.gat_ckpt} (epoch {start_epoch}, AUC {best_auc:.4f})")

    # ── Training loop ────────────────────────────────────────────────────
    epochs          = gat_cfg.training.epochs
    patience        = gat_cfg.training.early_stopping.patience
    patience_counter= 0
    history         = defaultdict(list)

    logger.info(f"Training for up to {epochs} epochs (patience={patience})")

    for epoch in range(start_epoch, epochs + 1):

        train_loss = train_epoch(gat_model, train_loader, optimizer, device, gat_cfg, epoch, logger)
        history["train_loss"].append(train_loss)

        val_metrics = evaluate_gnn_fast(gat_model, val_loader, device, gat_cfg)
        for k, v in val_metrics.items():
            history[k].append(v)

        logger.info(f"Epoch {epoch} Val: " +
                    ", ".join(f"{k}: {v:.4f}" for k, v in val_metrics.items()))

        # ── Checkpoint & early stopping ──────────────────────────────────
        current_auc = val_metrics.get("val_auc_roc", 0.0)
        if current_auc > best_auc:
            best_auc       = current_auc
            patience_counter = 0
            torch.save({
                "epoch":               epoch,
                "model_state_dict":    gat_model.state_dict(),
                "optimizer_state_dict":optimizer.state_dict(),
                "metrics":             val_metrics,
            }, checkpoint_dir / "best_gat_model.pth")
            logger.info(f"  ✓ New best saved (AUC={best_auc:.4f})")
        else:
            patience_counter += 1
            logger.info(f"  No improvement ({patience_counter}/{patience})")
            if patience_counter >= patience:
                logger.info("Early stopping triggered.")
                break

        # Save history every epoch
        with open(checkpoint_dir / "history.json", "w") as f:
            json.dump(dict(history), f, indent=2)

    logger.info(f"Training complete. Best AUC: {best_auc:.4f}")
    logger.info(f"Checkpoint: {checkpoint_dir / 'best_gat_model.pth'}")


if __name__ == "__main__":
    main()
