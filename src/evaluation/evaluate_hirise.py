import sys
import traceback
from pathlib import Path
import time
import pandas as pd
import torch
import numpy as np

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import load_config
from src.utils.seed import set_seed, get_device
from src.data.loaders.hirise_loader import HiRISEDataset
from src.inference.pipeline import PA_GNN_Pipeline


def evaluate_hirise_dataset(pipeline, dataset, baselines, device, hcr_threshold=0.70, max_samples=50):
    """
    HiRISE-specific evaluator.
    HiRISE has image-level risk labels (not pixel maps), so HCR is computed
    purely from the A* path node risk scores (no pixel GT comparison).
    """
    results = {b: {'success': 0, 'total': 0, 'hcr_list': [], 'time_list': []} for b in baselines}

    for i in range(min(max_samples, len(dataset))):
        img_tensor, risk_tensor, meta = dataset[i]   # risk_tensor: (1,) image-level scalar

        H, W = img_tensor.shape[1], img_tensor.shape[2]
        start = (int(H * 0.1), int(W * 0.1))
        goal  = (int(H * 0.9), int(W * 0.9))

        for bl in baselines:
            t0 = time.time()
            try:
                path, data, _, _ = pipeline.run(
                    img_tensor, start_coords=start, goal_coords=goal, run_baseline=bl
                )
            except Exception as e:
                print(f"  [WARN] sample {i} baseline {bl}: {e}")
                traceback.print_exc()
                results[bl]['total'] += 1
                results[bl]['time_list'].append(time.time() - t0)
                continue

            t_elapsed = time.time() - t0
            results[bl]['total'] += 1
            results[bl]['time_list'].append(t_elapsed)

            if path is not None:
                results[bl]['success'] += 1
                high_cost = sum(1 for p in path if p['risk'] > hcr_threshold)
                hcr = high_cost / len(path)
                results[bl]['hcr_list'].append(hcr)

        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{min(max_samples, len(dataset))} samples...")

    final = {}
    for bl in baselines:
        total = results[bl]['total']
        final[bl] = {
            'Success Rate': results[bl]['success'] / total if total > 0 else 0.0,
            'HCR':          np.mean(results[bl]['hcr_list']) if results[bl]['hcr_list'] else 1.0,
            'Time (s)':     np.mean(results[bl]['time_list']) if results[bl]['time_list'] else 0.0,
        }
    return final


def main():
    base_cfg   = load_config('configs/base.yaml')
    cnn_cfg    = load_config('configs/cnn/mobilenetv3.yaml')
    phys_cfg   = load_config('configs/physics.yaml')
    fusion_cfg = load_config('configs/fusion/adaptive_fusion.yaml')
    gat_cfg    = load_config('configs/gnn/gatv2.yaml')
    hirise_cfg = load_config('configs/datasets/hirise.yaml')

    device = get_device(base_cfg.project.device)
    set_seed(base_cfg.project.seed)

    results_dir = Path(base_cfg.paths.results) / "stage7_eval"
    results_dir.mkdir(parents=True, exist_ok=True)

    fusion_ckpt = Path(base_cfg.paths.checkpoints) / "fusion"   / "best_model.pth"
    gat_ckpt    = Path(base_cfg.paths.checkpoints) / "gnn_fast" / "best_gat_model.pth"
    print(f"Fusion ckpt : {'FOUND' if fusion_ckpt.exists() else 'NOT FOUND'} ({fusion_ckpt})")
    print(f"GATv2 ckpt  : {'FOUND' if gat_ckpt.exists() else 'NOT FOUND'} ({gat_ckpt})")

    pipeline = PA_GNN_Pipeline(
        base_cfg, cnn_cfg, phys_cfg, fusion_cfg, gat_cfg, device,
        fusion_ckpt=str(fusion_ckpt) if fusion_ckpt.exists() else None,
        gat_ckpt=str(gat_ckpt)    if gat_ckpt.exists()    else None,
    )

    test_ds  = HiRISEDataset.from_config(base_cfg, hirise_cfg)
    baselines = ['b1_euclidean', 'b2_physics', 'b3_learned', 'b4_static', 'proposed']

    print(f"Evaluating baselines on HiRISE (Cross-Domain)... [{len(test_ds)} samples, capped at 50]")
    hcr_thresh = getattr(gat_cfg.graph, 'inference_threshold', 0.70)
    results    = evaluate_hirise_dataset(pipeline, test_ds, baselines, device, hcr_threshold=hcr_thresh)

    df = pd.DataFrame.from_dict(results, orient='index')
    print("\n--- Final Results (HiRISE Cross-Domain) ---")
    print(df.to_string())

    df.to_csv(results_dir / "hirise_results.csv")
    print(f"\nResults saved to {results_dir / 'hirise_results.csv'}")


if __name__ == "__main__":
    main()
