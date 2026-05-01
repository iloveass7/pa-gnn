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

# Add project root
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import load_config
from src.utils.seed import set_seed, get_device
from src.data.loaders.ai4mars_loader import AI4MarsDataset
from src.models.fusion.fusion_model import EndToEndFusionModel
from src.models.fusion.adaptive_fusion import get_static_fusion
from src.evaluation.metrics import compute_metrics

@torch.no_grad()
def evaluate_and_visualize_fusion(model, dataloader, device, save_dir):
    model.eval()
    
    metrics = {
        'H_physics': {'hazard_recall': 0.0, 'iou': 0.0},
        'H_learned': {'hazard_recall': 0.0, 'iou': 0.0},
        'H_static': {'hazard_recall': 0.0, 'iou': 0.0},
        'H_final': {'hazard_recall': 0.0, 'iou': 0.0}
    }
    num_batches = len(dataloader)
    
    vis_data = []
    
    for i, (images, targets, _) in enumerate(dataloader):
        images = images.to(device)
        targets = targets.to(device)
        
        preds = model(images)
        h_phys = preds['h_physics']
        h_learn = preds['h_learned']
        alpha = preds['alpha']
        h_final = preds['h_final']
        
        h_static = get_static_fusion(h_phys, h_learn, alpha=0.5)
        
        # Metrics
        m_phys = compute_metrics(h_phys, targets)
        m_learn = compute_metrics(h_learn, targets)
        m_static = compute_metrics(h_static, targets)
        m_final = compute_metrics(h_final, targets)
        
        for k in ['hazard_recall', 'iou']:
            metrics['H_physics'][k] += m_phys[k]
            metrics['H_learned'][k] += m_learn[k]
            metrics['H_static'][k] += m_static[k]
            metrics['H_final'][k] += m_final[k]
            
        if len(vis_data) < 5:
            vis_data.append({
                'img': images[0].cpu(),
                'target': targets[0].cpu(),
                'h_phys': h_phys[0].cpu(),
                'h_learn': h_learn[0].cpu(),
                'alpha': alpha[0].cpu(),
                'h_final': h_final[0].cpu()
            })
            
    for approach in metrics:
        for k in metrics[approach]:
            metrics[approach][k] /= num_batches
            
    # Visualization: Side-by-side
    # Original | Target | H_physics | H_learned | Alpha | H_final
    fig, axes = plt.subplots(5, 6, figsize=(20, 16))
    risk_cmap = plt.cm.RdYlGn_r
    
    for i, item in enumerate(vis_data):
        img_np = item['img'].mean(dim=0).numpy()
        target = item['target'][0].numpy()
        target_masked = np.ma.masked_where(target < 0, target)
        
        axes[i, 0].imshow(img_np, cmap='gray')
        axes[i, 0].set_title("Original")
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(target_masked, cmap=risk_cmap, vmin=0, vmax=1)
        axes[i, 1].set_title("Ground Truth")
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(item['h_phys'][0].numpy(), cmap=risk_cmap, vmin=0, vmax=1)
        axes[i, 2].set_title("H_physics")
        axes[i, 2].axis('off')
        
        axes[i, 3].imshow(item['h_learn'][0].numpy(), cmap=risk_cmap, vmin=0, vmax=1)
        axes[i, 3].set_title("H_learned")
        axes[i, 3].axis('off')
        
        # Alpha: 0 (blue) = physics, 1 (red) = learned
        axes[i, 4].imshow(item['alpha'][0].numpy(), cmap='coolwarm', vmin=0, vmax=1)
        axes[i, 4].set_title("Alpha (Red=CNN)")
        axes[i, 4].axis('off')
        
        axes[i, 5].imshow(item['h_final'][0].numpy(), cmap=risk_cmap, vmin=0, vmax=1)
        axes[i, 5].set_title("H_final")
        axes[i, 5].axis('off')
        
    plt.tight_layout()
    out_path = save_dir / "fusion_comparison_heatmaps.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization to {out_path}")
    
    return metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_cfg', type=str, default='configs/base.yaml')
    parser.add_argument('--cnn_cfg', type=str, default='configs/cnn/mobilenetv3.yaml')
    parser.add_argument('--phys_cfg', type=str, default='configs/physics.yaml')
    parser.add_argument('--fusion_cfg', type=str, default='configs/fusion/adaptive_fusion.yaml')
    parser.add_argument('--fusion_ckpt', type=str, default='checkpoints/fusion/best_model.pth')
    args = parser.parse_args()

    base_cfg = load_config(args.base_cfg)
    cnn_cfg = load_config(args.cnn_cfg)
    phys_cfg = load_config(args.phys_cfg)
    fusion_cfg = load_config(args.fusion_cfg)
    
    set_seed(base_cfg.project.seed)
    device = get_device(base_cfg.project.device)
    
    results_dir = Path(base_cfg.paths.results) / "stage4"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    model = EndToEndFusionModel(cnn_cfg, phys_cfg, fusion_cfg, freeze_cnn=True).to(device)
    
    ckpt_path = Path(args.fusion_ckpt)
    if not ckpt_path.exists():
        print(f"Checkpoint not found at {ckpt_path}. Running with random weights to generate dummy visualizations.")
    else:
        print(f"Loading checkpoint from {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        
    # Data Loaders
    print("Loading datasets...")
    ai4mars_cfg = load_config("configs/datasets/ai4mars.yaml")
    test_ds = AI4MarsDataset.from_config(base_cfg, ai4mars_cfg, split="test")
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False)
    
    print("\nEvaluating Adaptive Fusion...")
    metrics = evaluate_and_visualize_fusion(model, test_loader, device, results_dir)
    
    print("\nFusion Metrics Comparison:")
    for approach, vals in metrics.items():
        print(f"  {approach}: Hazard Recall = {vals['hazard_recall']:.4f}, mIoU = {vals['iou']:.4f}")
        
    with open(results_dir / "fusion_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

if __name__ == "__main__":
    main()
