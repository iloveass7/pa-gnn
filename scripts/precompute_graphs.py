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
    data/graphs/val/graph_00000.pt
    data/graphs/test/graph_00000.pt
    data/graphs/{split}/manifest.json   <- index used by PrecomputedGraphDataset
"""

import sys
import argparse
import json
import time
from pathlib import Path
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import load_config
from src.utils.seed import set_seed, get_device
from src.utils.logger import get_logger, setup_logger
from src.data.loaders.ai4mars_loader import AI4MarsDataset
from src.models.fusion.fusion_model import EndToEndFusionModel
from src.models.gnn.graph_builder import GraphBuilder


def precompute_split(split, base_cfg, cnn_cfg, phys_cfg, fusion_cfg, gat_cfg,
                     fusion_model, graph_builder, device, out_root, logger):

    ai4mars_cfg = load_config(project_root / "configs/datasets/ai4mars.yaml")
    dataset = AI4MarsDataset.from_config(base_cfg, ai4mars_cfg, split=split)

    out_dir = Path(out_root) / split
    out_dir.mkdir(parents=True, exist_ok=True)

    # batch_size=1, no shuffle — keeps index <-> filename mapping stable
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    manifest = []
    skipped  = 0
    saved    = 0

    logger.info(f"[{split}] Processing {len(dataset)} images -> {out_dir}")
    t_start = time.time()

    fusion_model.eval()

    with torch.no_grad():
        for idx, (images, targets, meta) in enumerate(tqdm(loader, desc=f"[{split}]")):
            # images  shape: (1, C, H, W)
            # targets shape: (1, H, W)

            # Resume support — if file already exists from a previous interrupted
            # run, skip building it again and just re-add it to the manifest.
            fname = f"graph_{saved:05d}.pt"
            fpath = out_dir / fname
            if fpath.exists():
                stem = meta["stem"][0] if isinstance(meta["stem"], (list, tuple)) else meta["stem"]
                manifest.append({"idx": saved, "stem": stem, "file": fname})
                saved += 1
                continue

            images  = images.to(device)
            targets = targets.to(device)

            # ------------------------------------------------------------------
            # THE FIX: graph_builder.build() expects (C,H,W) and (H,W) — single
            # image tensors, not batched.  Use [0] to strip the batch dimension.
            #
            # DO NOT use .squeeze(0) — it is a no-op when dim-0 size != 1 and
            # would silently pass the wrong shape if batch_size were ever changed.
            # [0] always extracts the first element regardless of dim-0 size.
            # ------------------------------------------------------------------
            fusion_dict_batch = fusion_model(images)
            fusion_dict = {
                k: v[0] if isinstance(v, torch.Tensor) else v
                for k, v in fusion_dict_batch.items()
            }
            data = graph_builder.build(images[0], fusion_dict, target=targets[0])

            if data.x.size(0) == 0:
                # All-NULL tile — skip, do not write file.
                # The manifest won't contain this index so the training loader
                # will never try to load it.
                skipped += 1
                continue

            # Save on CPU so files are portable and don't pin VRAM
            data = data.cpu()

            stem = meta["stem"][0] if isinstance(meta["stem"], (list, tuple)) else meta["stem"]
            torch.save(data, fpath)
            manifest.append({"idx": saved, "stem": stem, "file": fname})
            saved += 1

            if saved % 500 == 0:
                elapsed = time.time() - t_start
                rate    = saved / (elapsed + 1e-8)
                eta_min = (len(dataset) - idx) / (rate + 1e-8) / 60
                logger.info(
                    f"  [{split}] {saved}/{len(dataset)} saved | "
                    f"{rate:.1f} img/s | ETA {eta_min:.0f} min"
                )

    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    elapsed = time.time() - t_start
    logger.info(
        f"[{split}] Done: {saved} saved, {skipped} empty-skipped | "
        f"Total: {elapsed/60:.1f} min"
    )
    return saved


def main():
    parser = argparse.ArgumentParser(description="Precompute PA-GNN graphs (run once)")
    parser.add_argument("--split",       type=str, default="all",
                        choices=["train", "val", "test", "all"])
    parser.add_argument("--base_cfg",    type=str, default="configs/base.yaml")
    parser.add_argument("--cnn_cfg",     type=str, default="configs/cnn/mobilenetv3.yaml")
    parser.add_argument("--phys_cfg",    type=str, default="configs/physics.yaml")
    parser.add_argument("--fusion_cfg",  type=str, default="configs/fusion/adaptive_fusion.yaml")
    parser.add_argument("--gat_cfg",     type=str, default="configs/gnn/gatv2.yaml")
    parser.add_argument("--fusion_ckpt", type=str, default=None,
                        help="Path to trained fusion checkpoint (.pth)")
    parser.add_argument("--out_dir",     type=str, default=None,
                        help="Root output folder (default: <paths.processed>/graphs)")
    args = parser.parse_args()

    base_cfg   = load_config(project_root / args.base_cfg)
    cnn_cfg    = load_config(project_root / args.cnn_cfg)
    phys_cfg   = load_config(project_root / args.phys_cfg)
    fusion_cfg = load_config(project_root / args.fusion_cfg)
    gat_cfg    = load_config(project_root / args.gat_cfg)

    set_seed(base_cfg.project.seed)
    device = get_device(base_cfg.project.device)

    log_dir = Path(base_cfg.paths.logs) / "precompute"
    log_dir.mkdir(parents=True, exist_ok=True)
    setup_logger(log_file=log_dir / "precompute_graphs.log")
    logger = get_logger("PrecomputeGraphs")

    out_root = args.out_dir or str(Path(base_cfg.paths.processed) / "graphs")
    logger.info(f"Output directory: {out_root}")

    # Load fusion model (always frozen here — we're just doing inference)
    logger.info("Loading fusion model...")
    fusion_model = EndToEndFusionModel(
        cnn_cfg, phys_cfg, fusion_cfg, freeze_cnn=True
    ).to(device)
    fusion_model.eval()

    if args.fusion_ckpt and Path(args.fusion_ckpt).exists():
        ckpt = torch.load(args.fusion_ckpt, map_location=device, weights_only=False)
        fusion_model.load_state_dict(ckpt["model_state_dict"])
        logger.info(f"Loaded fusion checkpoint: {args.fusion_ckpt}")
    else:
        logger.warning(
            "No fusion checkpoint provided — using untrained weights. "
            "Run train_fusion.py first, then re-run this script."
        )

    graph_builder = GraphBuilder(gat_cfg)
    splits = ["train", "val", "test"] if args.split == "all" else [args.split]

    total_saved = 0
    t0 = time.time()

    for split in splits:
        n = precompute_split(
            split, base_cfg, cnn_cfg, phys_cfg, fusion_cfg, gat_cfg,
            fusion_model, graph_builder, device, out_root, logger
        )
        total_saved += n

    elapsed = time.time() - t0
    logger.info(f"=== Done: {total_saved} total graphs in {elapsed/60:.1f} min ===")
    logger.info(f"Next: python scripts/train_gnn_fast.py --graphs_dir {out_root}")


if __name__ == "__main__":
    main()