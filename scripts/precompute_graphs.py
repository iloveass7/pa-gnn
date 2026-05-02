"""
precompute_graphs.py — One-time graph pre-computation script.

Runs the frozen fusion model over every AI4Mars image and saves the resulting
PyG Data object to disk as a .pt file. After this runs once (~3 hrs), the GNN
training script can load graphs in ~0.01s instead of building them in ~0.8s.

Usage:
    python scripts/precompute_graphs.py                     # all splits
    python scripts/precompute_graphs.py --split train       # train only
    python scripts/precompute_graphs.py --fusion_ckpt checkpoints/fusion/best_model.pth

Outputs:
    data/graphs/train/graph_00000.pt
    data/graphs/train/graph_00001.pt
    ...
    data/graphs/val/graph_00000.pt
    data/graphs/test/graph_00000.pt
    data/graphs/{split}/manifest.json   ← stems list, used by dataset class
"""

import sys
import argparse
import json
import time
from pathlib import Path
from tqdm import tqdm

import torch

# Add project root
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import load_config
from src.utils.seed import set_seed, get_device
from src.utils.logger import get_logger, setup_logger
from src.data.loaders.ai4mars_loader import AI4MarsDataset
from src.models.fusion.fusion_model import EndToEndFusionModel
from src.models.gnn.graph_builder import GraphBuilder
from torch.utils.data import DataLoader


def precompute_split(split, base_cfg, cnn_cfg, phys_cfg, fusion_cfg, gat_cfg,
                     fusion_model, graph_builder, device, out_root, logger):
    """
    Precompute and save graphs for one dataset split (train/val/test).
    Returns number of graphs saved.
    """
    ai4mars_cfg = load_config("configs/datasets/ai4mars.yaml")
    dataset = AI4MarsDataset.from_config(base_cfg, ai4mars_cfg, split=split)

    out_dir = Path(out_root) / split
    out_dir.mkdir(parents=True, exist_ok=True)

    # Use batch_size=1, no shuffle — we need deterministic idx↔file mapping
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    manifest = []   # list of {"idx": i, "stem": stem, "file": "graph_00000.pt"}
    skipped = 0
    saved = 0

    logger.info(f"[{split}] Processing {len(dataset)} images → {out_dir}")
    t_start = time.time()

    fusion_model.eval()

    with torch.no_grad():
        for idx, (images, targets, meta) in enumerate(tqdm(loader, desc=f"[{split}]")):
            images  = images.to(device)
            targets = targets.to(device)

            # Build graph (this is the slow step — ~0.8s per image on CPU)
            fusion_dict = fusion_model(images)
            data = graph_builder.build(images, fusion_dict, target=targets)

            if data.x.size(0) == 0:
                skipped += 1
                continue

            # Move everything to CPU before saving — saves VRAM and keeps files portable
            data = data.cpu()

            fname  = f"graph_{saved:05d}.pt"
            fpath  = out_dir / fname
            stem   = meta["stem"][0] if isinstance(meta["stem"], (list, tuple)) else meta["stem"]

            torch.save(data, fpath)

            manifest.append({
                "idx":  saved,
                "stem": stem,
                "file": fname,
            })
            saved += 1

            if saved % 500 == 0:
                elapsed = time.time() - t_start
                rate = saved / elapsed
                eta  = (len(dataset) - saved) / (rate + 1e-8)
                logger.info(f"  [{split}] {saved}/{len(dataset)} saved | "
                            f"{rate:.1f} graphs/s | ETA {eta/60:.0f} min")

    # Save manifest
    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    elapsed = time.time() - t_start
    logger.info(f"[{split}] Done: {saved} graphs saved, {skipped} skipped | "
                f"Total time: {elapsed/60:.1f} min | {out_dir}")

    return saved


def main():
    parser = argparse.ArgumentParser(description="Precompute PA-GNN graphs (one-time)")
    parser.add_argument("--split",       type=str, default="all",
                        help="Which split(s) to precompute: train | val | test | all")
    parser.add_argument("--base_cfg",    type=str, default="configs/base.yaml")
    parser.add_argument("--cnn_cfg",     type=str, default="configs/cnn/mobilenetv3.yaml")
    parser.add_argument("--phys_cfg",    type=str, default="configs/physics.yaml")
    parser.add_argument("--fusion_cfg",  type=str, default="configs/fusion/adaptive_fusion.yaml")
    parser.add_argument("--gat_cfg",     type=str, default="configs/gnn/gatv2.yaml")
    parser.add_argument("--fusion_ckpt", type=str, default=None,
                        help="Path to trained fusion model checkpoint (.pth)")
    parser.add_argument("--out_dir",     type=str, default=None,
                        help="Output directory for .pt files (default: <processed>/graphs)")
    args = parser.parse_args()

    base_cfg   = load_config(args.base_cfg)
    cnn_cfg    = load_config(args.cnn_cfg)
    phys_cfg   = load_config(args.phys_cfg)
    fusion_cfg = load_config(args.fusion_cfg)
    gat_cfg    = load_config(args.gat_cfg)

    set_seed(base_cfg.project.seed)
    device = get_device(base_cfg.project.device)

    log_dir = Path(base_cfg.paths.logs) / "precompute"
    log_dir.mkdir(parents=True, exist_ok=True)
    setup_logger(log_file=log_dir / "precompute_graphs.log")
    logger = get_logger("PrecomputeGraphs")

    # Output root
    out_root = args.out_dir if args.out_dir else str(Path(base_cfg.paths.processed) / "graphs")
    logger.info(f"Output directory: {out_root}")

    # Load fusion model — always frozen during precompute
    logger.info("Loading fusion model...")
    fusion_model = EndToEndFusionModel(
        cnn_cfg, phys_cfg, fusion_cfg, freeze_cnn=True
    ).to(device)
    fusion_model.eval()

    if args.fusion_ckpt and Path(args.fusion_ckpt).exists():
        ckpt = torch.load(args.fusion_ckpt, map_location=device, weights_only=True)
        fusion_model.load_state_dict(ckpt["model_state_dict"])
        logger.info(f"Loaded fusion checkpoint: {args.fusion_ckpt}")
    else:
        logger.warning("No fusion checkpoint provided — using untrained weights. "
                       "Run train_fusion.py first, then precompute.")

    graph_builder = GraphBuilder(gat_cfg)

    # Determine which splits to process
    splits = ["train", "val", "test"] if args.split == "all" else [args.split]

    total_saved = 0
    t_global = time.time()

    for split in splits:
        n = precompute_split(
            split, base_cfg, cnn_cfg, phys_cfg, fusion_cfg, gat_cfg,
            fusion_model, graph_builder, device, out_root, logger
        )
        total_saved += n

    elapsed = time.time() - t_global
    logger.info(f"=== Precomputation complete: {total_saved} graphs in {elapsed/60:.1f} min ===")
    logger.info(f"Next step: python scripts/train_gnn_fast.py --graphs_dir {out_root}")


if __name__ == "__main__":
    main()
