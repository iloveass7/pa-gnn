import sys
import argparse
from pathlib import Path
import time
import json
import pandas as pd

import torch
import numpy as np

# Add project root
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import load_config
from src.utils.seed import set_seed, get_device
from src.data.loaders.ai4mars_loader import AI4MarsDataset
from src.inference.pipeline import PA_GNN_Pipeline

def evaluate_dataset(pipeline, dataset, baselines, device):
    results = {b: {'success': 0, 'total': 0, 'hcr_list': [], 'time_list': []} for b in baselines}
    
    for i in range(len(dataset)):
        if i >= 50: break # Subset for reasonable script duration
        
        img_tensor, target, meta = dataset[i]
        
        H, W = img_tensor.shape[1], img_tensor.shape[2]
        start = (int(H*0.1), int(W*0.1))
        goal = (int(H*0.9), int(W*0.9))
        
        for bl in baselines:
            t0 = time.time()
            path, data, _, _ = pipeline.run(img_tensor, start_coords=start, goal_coords=goal, run_baseline=bl)
            t_elapsed = time.time() - t0
            
            results[bl]['total'] += 1
            results[bl]['time_list'].append(t_elapsed)
            
            if path is not None:
                results[bl]['success'] += 1
                
                high_cost_nodes = sum(1 for p in path if p['risk'] > 0.7)
                hcr = high_cost_nodes / len(path)
                results[bl]['hcr_list'].append(hcr)
                
    final_results = {}
    for bl in baselines:
        succ_rate = results[bl]['success'] / results[bl]['total'] if results[bl]['total'] > 0 else 0
        mean_hcr = np.mean(results[bl]['hcr_list']) if len(results[bl]['hcr_list']) > 0 else 1.0
        mean_time = np.mean(results[bl]['time_list']) if len(results[bl]['time_list']) > 0 else 0.0
        
        final_results[bl] = {
            'Success Rate': succ_rate,
            'HCR': mean_hcr,
            'Time (s)': mean_time
        }
        
    return final_results

def main():
    base_cfg = load_config('configs/base.yaml')
    cnn_cfg = load_config('configs/cnn/mobilenetv3.yaml')
    phys_cfg = load_config('configs/physics.yaml')
    fusion_cfg = load_config('configs/fusion/adaptive_fusion.yaml')
    gat_cfg = load_config('configs/gnn/gatv2.yaml')
    ai4mars_cfg = load_config('configs/datasets/ai4mars.yaml')
    
    device = get_device(base_cfg.project.device)
    set_seed(base_cfg.project.seed)
    
    results_dir = Path(base_cfg.paths.results) / "stage7_eval"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    pipeline = PA_GNN_Pipeline(base_cfg, cnn_cfg, phys_cfg, fusion_cfg, gat_cfg, device)
    
    test_ds = AI4MarsDataset.from_config(base_cfg, ai4mars_cfg, split="test")
    
    baselines = ['b1_euclidean', 'b2_physics', 'b3_learned', 'b4_static', 'proposed']
    
    print("Evaluating baselines on AI4Mars Test Set...")
    results = evaluate_dataset(pipeline, test_ds, baselines, device)
    
    df = pd.DataFrame.from_dict(results, orient='index')
    print("\n--- Final Results (AI4Mars) ---")
    print(df.to_string())
    
    df.to_csv(results_dir / "ai4mars_results.csv")
    print(f"\nResults saved to {results_dir / 'ai4mars_results.csv'}")

if __name__ == "__main__":
    main()
