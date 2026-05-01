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
from src.data.loaders.hirise_loader import HiRISEDataset
from src.inference.pipeline import PA_GNN_Pipeline
from src.evaluation.evaluate_ai4mars import evaluate_dataset

def main():
    base_cfg = load_config('configs/base.yaml')
    cnn_cfg = load_config('configs/cnn/mobilenetv3.yaml')
    phys_cfg = load_config('configs/physics.yaml')
    fusion_cfg = load_config('configs/fusion/adaptive_fusion.yaml')
    gat_cfg = load_config('configs/gnn/gatv2.yaml')
    hirise_cfg = load_config('configs/datasets/hirise.yaml')
    
    device = get_device(base_cfg.project.device)
    set_seed(base_cfg.project.seed)
    
    results_dir = Path(base_cfg.paths.results) / "stage7_eval"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    pipeline = PA_GNN_Pipeline(base_cfg, cnn_cfg, phys_cfg, fusion_cfg, gat_cfg, device)
    
    test_ds = HiRISEDataset.from_config(base_cfg, hirise_cfg)
    
    baselines = ['b1_euclidean', 'b2_physics', 'b3_learned', 'b4_static', 'proposed']
    
    print("Evaluating baselines on HiRISE (Cross-Domain)...")
    results = evaluate_dataset(pipeline, test_ds, baselines, device)
    
    df = pd.DataFrame.from_dict(results, orient='index')
    print("\n--- Final Results (HiRISE) ---")
    print(df.to_string())
    
    df.to_csv(results_dir / "hirise_results.csv")
    print(f"\nResults saved to {results_dir / 'hirise_results.csv'}")

if __name__ == "__main__":
    main()
