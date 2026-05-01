import sys
import argparse
from pathlib import Path
import json

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score

# Add project root
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import load_config
from src.utils.seed import set_seed, get_device
from src.data.loaders.ai4mars_loader import AI4MarsDataset
from src.models.fusion.fusion_model import EndToEndFusionModel
from src.models.gnn.graph_builder import GraphBuilder
from src.models.gnn.gatv2 import PAGATv2
from src.training.weak_labels import compute_weak_labels

def visualize_projection(image, label_map, node_risks, save_path, title=""):
    """
    Project node-level risks back to the image resolution.
    """
    img_np = image.mean(dim=0).cpu().numpy()
    label_map_np = label_map.cpu().numpy()
    
    # Create projected risk map
    risk_map = np.zeros_like(img_np, dtype=np.float32)
    N = len(node_risks)
    
    for i in range(N):
        risk_map[label_map_np == i] = node_risks[i]
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    ax1.imshow(img_np, cmap='gray')
    ax1.set_title(f"{title} Original Image")
    ax1.axis('off')
    
    # Overlay risk
    ax2.imshow(img_np, cmap='gray')
    im = ax2.imshow(risk_map, cmap=plt.cm.RdYlGn_r, alpha=0.6, vmin=0, vmax=1)
    ax2.set_title(f"{title} Node Risk Projection")
    ax2.axis('off')
    
    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_cfg', type=str, default='configs/base.yaml')
    parser.add_argument('--cnn_cfg', type=str, default='configs/cnn/mobilenetv3.yaml')
    parser.add_argument('--phys_cfg', type=str, default='configs/physics.yaml')
    parser.add_argument('--fusion_cfg', type=str, default='configs/fusion/adaptive_fusion.yaml')
    parser.add_argument('--gat_cfg', type=str, default='configs/gnn/gatv2.yaml')
    parser.add_argument('--gat_ckpt', type=str, default='checkpoints/gnn/best_gat_model.pth')
    args = parser.parse_args()

    base_cfg = load_config(args.base_cfg)
    cnn_cfg = load_config(args.cnn_cfg)
    phys_cfg = load_config(args.phys_cfg)
    fusion_cfg = load_config(args.fusion_cfg)
    gat_cfg = load_config(args.gat_cfg)
    
    set_seed(base_cfg.project.seed)
    device = get_device(base_cfg.project.device)
    
    results_dir = Path(base_cfg.paths.results) / "stage6"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Models
    fusion_model = EndToEndFusionModel(cnn_cfg, phys_cfg, fusion_cfg, freeze_cnn=True).to(device)
    graph_builder = GraphBuilder(gat_cfg)
    gat_model = PAGATv2(gat_cfg).to(device)
    
    ckpt_path = Path(args.gat_ckpt)
    if ckpt_path.exists():
        print(f"Loading GATv2 checkpoint from {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
        gat_model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print(f"Checkpoint not found at {ckpt_path}. Running with random weights.")
        
    gat_model.eval()
    
    # Data Loaders
    print("Loading datasets...")
    ai4mars_cfg = load_config("configs/datasets/ai4mars.yaml")
    test_ds = AI4MarsDataset.from_config(base_cfg, ai4mars_cfg, split="test")
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    
    all_targets = []
    all_preds = []
    
    print("\nEvaluating GATv2 on AI4Mars Test Set...")
    for i, (images, targets, _) in enumerate(test_loader):
        images, targets = images.to(device), targets.to(device)
        
        with torch.no_grad():
            fusion_dict = fusion_model(images)
            data = graph_builder.build(images, fusion_dict, target=targets)
            
            if data.x.size(0) == 0: continue
            
            # Predict
            preds = gat_model(data.x, data.edge_index, data.edge_attr)
            
            # Metric aggregation
            valid_mask = (data.y >= 0) & data.active_mask
            if valid_mask.sum() > 0:
                valid_preds = preds[valid_mask].cpu().numpy()
                valid_targets = data.y[valid_mask].cpu().numpy()
                hard_targets = (valid_targets > 0.5).astype(np.float32)
                
                all_targets.extend(hard_targets)
                all_preds.extend(valid_preds)
            
            # 1. Visualize weak labeling (First image only)
            if i == 0:
                print("Generating Weak Labeling Visualization...")
                # Binarize GT for weak labeling logic
                hard_labels = (data.y > 0.5).float()
                hard_labels[data.y < 0] = -1.0
                
                updated_labels = compute_weak_labels(
                    data.edge_index, 
                    hard_labels, 
                    data.active_mask,
                    hops=gat_cfg.training.weak_labeling.hops,
                    weak_value=gat_cfg.training.weak_labeling.label_value
                )
                
                visualize_projection(images[0], data.label_map, updated_labels.cpu().numpy(), 
                                     results_dir / "weak_labels_projection.png", title="Weak Labels")
            
            # 2. Visualize final node risks
            if i < 5:
                # Fill in deactivated nodes with 1.0 (obstacle)
                full_preds = torch.ones_like(preds)
                full_preds[data.active_mask] = preds[data.active_mask]
                
                visualize_projection(images[0], data.label_map, full_preds.cpu().numpy(), 
                                     results_dir / f"gatv2_risk_ai4mars_{i}.png", title=f"AI4Mars[{i}]")
                                     
        if i >= 10: break # for verification, limit to 10 images
        
    # Metrics
    if len(all_targets) > 0:
        t = np.array(all_targets)
        p = np.array(all_preds)
        p_hard = (p > 0.5).astype(np.float32)
        
        auc = roc_auc_score(t, p) if len(np.unique(t)) > 1 else 0.5
        acc = accuracy_score(t, p_hard)
        prec = precision_score(t, p_hard, zero_division=0)
        rec = recall_score(t, p_hard, zero_division=0)
        
        metrics = {
            'auc_roc': auc,
            'accuracy': acc,
            'precision': prec,
            'hazard_recall': rec
        }
        
        print("\nGATv2 Node-Level Metrics (Test Set subset):")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
            
        with open(results_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

if __name__ == "__main__":
    main()
