"""
evaluate_ai4mars.py — Full evaluation of PA-GNN pipeline on AI4Mars test set.

Fixes applied from GNN Research Audit:
  m1: --max_samples CLI arg (was hardcoded 50 — only 15% of test set)
  m2: --n_trials random start/goal pairs per image (was fixed diagonal)
  C3: b5_no_gnn and oracle baselines added
  C6: --seeds multi-seed evaluation with mean±std and Wilcoxon signed-rank test
  PLR: Path Length Ratio metric added (path_length / straight_line_distance)
  C2: Logs when adaptive threshold relaxation was used

Usage:
    # Full evaluation (all 322 test samples, 3 random trials, 5 seeds):
    python src/evaluation/evaluate_ai4mars.py

    # Quick smoke-test (50 samples, 1 trial, 1 seed):
    python src/evaluation/evaluate_ai4mars.py --max_samples 50 --n_trials 1 --seeds 42
"""

import sys
import argparse
from pathlib import Path
import time
import json
import warnings

import torch
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import load_config
from src.utils.seed import set_seed, get_device
from src.data.loaders.ai4mars_loader import AI4MarsDataset
from src.inference.pipeline import PA_GNN_Pipeline


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_path_length(path):
    """Sum of Euclidean distances between consecutive node positions."""
    if path is None or len(path) < 2:
        return 0.0
    total = 0.0
    for i in range(len(path) - 1):
        p1 = np.array(path[i]['pos'])
        p2 = np.array(path[i + 1]['pos'])
        total += float(np.linalg.norm(p2 - p1))
    return total


def compute_plr(path):
    """
    Path Length Ratio = actual_path_length / straight_line_distance.
    PLR = 1.0 is perfect (straight line). Higher = more detour taken.
    Blueprint target: PLR < 1.30.
    """
    if path is None or len(path) < 2:
        return float('nan')
    straight = path[0].get('straight_line', None)
    if straight is None or straight < 1e-6:
        return float('nan')
    actual = compute_path_length(path)
    return actual / straight


# ── Per-sample evaluation ─────────────────────────────────────────────────────

def evaluate_one_image(pipeline, img_tensor, target, baselines, hcr_threshold,
                       n_trials, rng):
    """
    Evaluate all baselines on one image with n_trials random start/goal pairs.
    Returns dict: baseline → list of per-trial result dicts.
    """
    H, W = img_tensor.shape[1], img_tensor.shape[2]
    margin = 0.10  # keep start/goal at least 10% from edge

    results = {bl: [] for bl in baselines}

    for _ in range(n_trials):
        # m2: random start/goal (not fixed diagonal)
        sy = rng.integers(int(H * margin), int(H * (1 - margin)))
        sx = rng.integers(int(W * margin), int(W * (1 - margin)))
        gy = rng.integers(int(H * margin), int(H * (1 - margin)))
        gx = rng.integers(int(W * margin), int(W * (1 - margin)))

        # Ensure start ≠ goal (at least 30% of image diagonal apart)
        min_dist = 0.30 * np.sqrt(H**2 + W**2)
        attempts = 0
        while np.linalg.norm([sy - gy, sx - gx]) < min_dist and attempts < 10:
            gy = rng.integers(int(H * margin), int(H * (1 - margin)))
            gx = rng.integers(int(W * margin), int(W * (1 - margin)))
            attempts += 1

        start = (int(sy), int(sx))
        goal  = (int(gy), int(gx))

        for bl in baselines:
            t0 = time.time()
            try:
                path, _, _, _ = pipeline.run(
                    img_tensor,
                    start_coords=start,
                    goal_coords=goal,
                    run_baseline=bl,
                    ground_truth=target,   # needed by oracle baseline
                )
            except Exception as e:
                warnings.warn(f"Pipeline error on baseline={bl}: {e}")
                path = None
            elapsed = time.time() - t0

            success = path is not None
            hcr     = float(np.mean([s['risk'] > hcr_threshold
                                     for s in path])) if success else 1.0
            plr     = compute_plr(path)
            relaxed = any(s.get('relaxed_threshold') is not None
                          for s in path) if success else False

            results[bl].append({
                'success':  success,
                'hcr':      hcr,
                'plr':      plr,
                'time':     elapsed,
                'relaxed':  relaxed,
            })

    return results


# ── Dataset-level evaluation ───────────────────────────────────────────────────

def evaluate_dataset(pipeline, dataset, baselines, hcr_threshold,
                     max_samples, n_trials, seed):
    """Evaluate all baselines on up to max_samples images with given seed."""
    rng = np.random.default_rng(seed)
    n   = min(max_samples, len(dataset))

    # per_bl[baseline] = list of per-trial dicts
    per_bl = {bl: [] for bl in baselines}

    for i in range(n):
        img_tensor, target, _ = dataset[i]

        trial_results = evaluate_one_image(
            pipeline, img_tensor, target, baselines,
            hcr_threshold, n_trials, rng,
        )
        for bl in baselines:
            per_bl[bl].extend(trial_results[bl])

        if (i + 1) % 25 == 0:
            print(f"  [{seed}] {i+1}/{n} images done")

    # Aggregate
    agg = {}
    for bl in baselines:
        trials      = per_bl[bl]
        success     = [t['success'] for t in trials]
        hcr_vals    = [t['hcr']     for t in trials if t['success']]
        plr_vals    = [t['plr']     for t in trials
                       if t['success'] and not np.isnan(t['plr'])]
        times       = [t['time']    for t in trials]
        relaxed_pct = np.mean([t['relaxed'] for t in trials if t['success']]) \
                      if any(t['success'] for t in trials) else 0.0

        agg[bl] = {
            'success_rate': float(np.mean(success)),
            'hcr':          float(np.mean(hcr_vals))   if hcr_vals else 1.0,
            'plr':          float(np.mean(plr_vals))   if plr_vals else float('nan'),
            'time':         float(np.mean(times)),
            'relaxed_pct':  float(relaxed_pct),
            # raw arrays for cross-seed std + Wilcoxon
            '_hcr_raw':     hcr_vals,
            '_plr_raw':     plr_vals,
            '_success_raw': success,
        }
    return agg


# ── Statistical summary ────────────────────────────────────────────────────────

def aggregate_seeds(seed_results, baselines):
    """
    Combine per-seed results into mean ± std.
    Also runs paired Wilcoxon signed-rank test: proposed vs each other baseline.
    """
    summary = {}
    for bl in baselines:
        sr   = [r[bl]['success_rate'] for r in seed_results]
        hcr  = [r[bl]['hcr']         for r in seed_results]
        plr  = [r[bl]['plr']         for r in seed_results if not np.isnan(r[bl]['plr'])]
        t    = [r[bl]['time']        for r in seed_results]
        rx   = [r[bl]['relaxed_pct'] for r in seed_results]

        summary[bl] = {
            'Success Rate': f"{np.mean(sr):.3f} ± {np.std(sr):.3f}",
            'HCR':          f"{np.mean(hcr):.4f} ± {np.std(hcr):.4f}",
            'PLR':          f"{np.mean(plr):.3f} ± {np.std(plr):.3f}" if plr else "nan",
            'Time (s)':     f"{np.mean(t):.3f}",
            'Relaxed%':     f"{np.mean(rx)*100:.1f}%" if bl == 'proposed' else "—",
        }

    # Wilcoxon signed-rank test: proposed vs each other baseline on per-trial HCR.
    # Skipped gracefully if 'proposed' is not in the evaluated baseline set.
    wilcoxon_results = {}
    if 'proposed' not in baselines:
        return summary, wilcoxon_results

    proposed_hcr = seed_results[0]['proposed']['_hcr_raw']
    for bl in baselines:
        if bl == 'proposed':
            continue
        other_hcr = seed_results[0][bl]['_hcr_raw']
        n_pairs   = min(len(proposed_hcr), len(other_hcr))
        if n_pairs >= 10:
            try:
                stat, p = wilcoxon(proposed_hcr[:n_pairs], other_hcr[:n_pairs],
                                   alternative='less')   # proposed < other (lower HCR = better)
                wilcoxon_results[bl] = {'statistic': float(stat), 'p_value': float(p),
                                        'significant': p < 0.05}
            except Exception:
                wilcoxon_results[bl] = {'p_value': float('nan'), 'significant': False}

    return summary, wilcoxon_results


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate PA-GNN pipeline on AI4Mars test set"
    )
    parser.add_argument("--max_samples", type=int, default=322,
                        help="Max test images to evaluate (default: 322 = full set)")
    parser.add_argument("--n_trials",    type=int, default=3,
                        help="Random start/goal trials per image (default: 3)")
    parser.add_argument("--seeds",       type=int, nargs="+", default=[42, 123, 7],
                        help="RNG seeds for multi-seed evaluation (default: 42 123 7)")
    parser.add_argument("--baselines",   type=str, nargs="+",
                        default=["b1_euclidean", "b2_physics", "b3_learned",
                                 "b4_static", "b5_no_gnn", "proposed", "oracle"],
                        help="Baselines to evaluate")
    parser.add_argument("--no_oracle",   action="store_true",
                        help="Skip oracle baseline (no ground-truth needed)")
    args = parser.parse_args()

    if args.no_oracle and "oracle" in args.baselines:
        args.baselines.remove("oracle")

    base_cfg   = load_config("configs/base.yaml")
    cnn_cfg    = load_config("configs/cnn/mobilenetv3.yaml")
    phys_cfg   = load_config("configs/physics.yaml")
    fusion_cfg = load_config("configs/fusion/adaptive_fusion.yaml")
    gat_cfg    = load_config("configs/gnn/gatv2.yaml")
    ai4mars_cfg= load_config("configs/datasets/ai4mars.yaml")

    device = get_device(base_cfg.project.device)
    set_seed(base_cfg.project.seed)

    results_dir = Path(base_cfg.paths.results) / "stage7_eval"
    results_dir.mkdir(parents=True, exist_ok=True)

    fusion_ckpt = Path(base_cfg.paths.checkpoints) / "fusion"    / "best_model.pth"
    gat_ckpt    = Path(base_cfg.paths.checkpoints) / "gnn_fast"  / "best_gat_model.pth"

    print(f"Fusion ckpt : {'FOUND' if fusion_ckpt.exists() else 'MISSING'} ({fusion_ckpt})")
    print(f"GATv2 ckpt  : {'FOUND' if gat_ckpt.exists() else 'MISSING'} ({gat_ckpt})")

    pipeline = PA_GNN_Pipeline(
        base_cfg, cnn_cfg, phys_cfg, fusion_cfg, gat_cfg, device,
        fusion_ckpt=str(fusion_ckpt) if fusion_ckpt.exists() else None,
        gat_ckpt   =str(gat_ckpt)    if gat_ckpt.exists()    else None,
    )

    test_ds     = AI4MarsDataset.from_config(base_cfg, ai4mars_cfg, split="test")
    hcr_thresh  = float(gat_cfg.graph.get("inference_threshold", 0.70))
    baselines   = args.baselines

    print(f"\nEvaluating {args.max_samples} samples × {args.n_trials} trials "
          f"× {len(args.seeds)} seeds on AI4Mars test set...")
    print(f"Baselines: {baselines}\n")

    seed_results = []
    for seed in args.seeds:
        print(f"── Seed {seed} ──────────────────────────────────")
        agg = evaluate_dataset(
            pipeline, test_ds, baselines,
            hcr_threshold=hcr_thresh,
            max_samples=args.max_samples,
            n_trials=args.n_trials,
            seed=seed,
        )
        seed_results.append(agg)

    # ── Final summary ─────────────────────────────────────────────────────
    summary, wilcoxon_res = aggregate_seeds(seed_results, baselines)

    df = pd.DataFrame.from_dict(summary, orient="index")
    print("\n─── Final Results (AI4Mars) — mean ± std across seeds ───")
    print(df.to_string())

    print("\n─── Wilcoxon Signed-Rank Test: Proposed vs Baselines (HCR) ───")
    for bl, res in wilcoxon_res.items():
        sig = "✓ significant" if res.get("significant") else "✗ not significant"
        p   = res.get("p_value", float("nan"))
        print(f"  proposed < {bl:15s}  p={p:.4f}  {sig}")

    # Blueprint alignment check
    print("\n─── Blueprint Target Check ─────────────────────────────────────")
    proposed_sr  = float(np.mean([r['proposed']['success_rate'] for r in seed_results]))
    proposed_hcr = float(np.mean([r['proposed']['hcr']         for r in seed_results]))
    proposed_plr_vals = [r['proposed']['plr'] for r in seed_results
                         if not np.isnan(r['proposed']['plr'])]
    proposed_plr = float(np.mean(proposed_plr_vals)) if proposed_plr_vals else float('nan')

    print(f"  Success Rate  : {proposed_sr:.3f}  "
          f"{'✓' if proposed_sr >= 0.95 else '✗ (target: >0.95)'}")
    print(f"  HCR           : {proposed_hcr:.4f}  "
          f"{'✓' if proposed_hcr <= 0.05 else '✗ (target: <0.05)'}")
    print(f"  PLR           : {proposed_plr:.3f}  "
          f"{'✓' if proposed_plr <= 1.30 else '✗ (target: <1.30)'}"
          if not np.isnan(proposed_plr) else "  PLR           : nan")

    # Save
    df.to_csv(results_dir / "ai4mars_results.csv")
    with open(results_dir / "wilcoxon.json", "w") as f:
        json.dump(wilcoxon_res, f, indent=2)
    with open(results_dir / "eval_config.json", "w") as f:
        json.dump({
            "max_samples": args.max_samples,
            "n_trials":    args.n_trials,
            "seeds":       args.seeds,
            "baselines":   baselines,
            "hcr_threshold": hcr_thresh,
        }, f, indent=2)

    print(f"\nResults saved to {results_dir}/")


if __name__ == "__main__":
    main()

