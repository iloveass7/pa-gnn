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
from src.data.loaders.hirise_loader import HiRISEDataset
from src.data.loaders.ctx_loader import CTXDataset
from src.models.cnn.risk_model import RiskModel
from src.evaluation.metrics import compute_metrics

def plot_training_curves(history_path, save_dir):
    if not history_path.exists():
        print(f"History file not found at {history_path}")
        return
        
    with open(history_path, 'r') as f:
        history = json.load(f)
        
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(epochs, history['train_loss'], label='Train Loss')
    ax1.plot(epochs, history['val_loss'], label='Val Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # Metric plot
    if 'val_hazard_recall' in history:
        ax2.plot(epochs, history['val_hazard_recall'], label='Val Hazard Recall')
    if 'val_iou' in history:
        ax2.plot(epochs, history['val_iou'], label='Val mIoU')
    ax2.set_title('Validation Metrics')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Score')
    ax2.legend()
    
    out_path = save_dir / "cnn_training_curves.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved training curves to {out_path}")

@torch.no_grad()
def evaluate_ai4mars(model, dataloader, device, save_dir):
    model.eval()
    all_metrics = {'hazard_recall': 0.0, 'iou': 0.0, 'ece': 0.0}
    num_batches = len(dataloader)
    
    # For visualizations
    vis_images, vis_preds, vis_targets = [], [], []
    
    for i, (images, targets, _) in enumerate(dataloader):
        images = images.to(device)
        targets = targets.to(device)
        preds = model(images)
        
        batch_metrics = compute_metrics(preds, targets)
        for k in all_metrics:
            all_metrics[k] += batch_metrics[k]
            
        if len(vis_images) < 5:
            # Take first image from batch
            vis_images.append(images[0].cpu())
            vis_preds.append(preds[0].cpu())
            vis_targets.append(targets[0].cpu())
            
    for k in all_metrics:
        all_metrics[k] /= num_batches
        
    # Plot visualizations
    fig, axes = plt.subplots(5, 3, figsize=(12, 18))
    risk_cmap = plt.cm.RdYlGn_r
    
    for i in range(len(vis_images)):
        img = vis_images[i].mean(dim=0).numpy() # Grayscale
        pred = vis_preds[i][0].numpy()
        target = vis_targets[i][0].numpy()
        
        target_masked = np.ma.masked_where(target < 0, target)
        
        axes[i, 0].imshow(img, cmap='gray')
        axes[i, 0].set_title("Input Image")
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(target_masked, cmap=risk_cmap, vmin=0, vmax=1)
        axes[i, 1].set_title("Ground Truth Risk")
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(pred, cmap=risk_cmap, vmin=0, vmax=1)
        axes[i, 2].set_title("H_learned (Predicted)")
        axes[i, 2].axis('off')
        
    plt.tight_layout()
    out_path = save_dir / "ai4mars_test_heatmaps.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved AI4Mars heatmaps to {out_path}")
    
    return all_metrics

@torch.no_grad()
def evaluate_hirise(model, dataloader, device):
    model.eval()
    
    class_correct = {}
    class_total = {}
    
    for images, targets, meta in dataloader:
        images = images.to(device)
        # targets are image-level, shape (B, 1)
        preds = model(images) # shape (B, 1, 512, 512)
        
        # Aggregate patch-level risk to image-level risk (e.g. mean or 90th percentile)
        # Let's use 90th percentile of predicted risk as the image-level risk
        for i in range(images.size(0)):
            p = preds[i, 0].cpu().numpy()
            img_risk = np.percentile(p, 90)
            
            t_risk = meta['risk_score'][i].item()
            cls_name = meta['class_name'][i]
            
            if cls_name not in class_correct:
                class_correct[cls_name] = 0
                class_total[cls_name] = 0
                
            class_total[cls_name] += 1
            
            # Simple thresholding for accuracy
            is_hazardous_gt = t_risk > 0.7
            is_hazardous_pred = img_risk > 0.5
            if is_hazardous_gt == is_hazardous_pred:
                class_correct[cls_name] += 1
                
    results = {}
    total_acc = 0
    total_count = sum(class_total.values())
    
    for cls_name in class_total:
        acc = class_correct[cls_name] / class_total[cls_name]
        results[cls_name] = acc
        total_acc += class_correct[cls_name]
        
    results['overall_accuracy'] = total_acc / total_count if total_count > 0 else 0
    return results

@torch.no_grad()
def evaluate_ctx(model, dataloader, device, save_dir, base_cfg):
    model.eval()
    
    indices = dataloader.dataset.select_demo_tiles(n=3, seed=base_cfg.project.seed)
    
    fig, axes = plt.subplots(3, 2, figsize=(10, 12))
    risk_cmap = plt.cm.RdYlGn_r
    
    for i, idx in enumerate(indices):
        img, meta = dataloader.dataset[idx]
        img_tensor = img.unsqueeze(0).to(device)
        
        pred = model(img_tensor)[0, 0].cpu().numpy()
        img_np = img.mean(dim=0).numpy()
        
        axes[i, 0].imshow(img_np, cmap='gray')
        axes[i, 0].set_title(f"CTX Input: {meta['filename'][:15]}")
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(pred, cmap=risk_cmap, vmin=0, vmax=1)
        axes[i, 1].set_title("H_learned (Predicted)")
        axes[i, 1].axis('off')
        
    plt.tight_layout()
    out_path = save_dir / "ctx_demo_heatmaps.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved CTX heatmaps to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_cfg', type=str, default='configs/base.yaml')
    parser.add_argument('--cnn_cfg', type=str, default='configs/cnn/mobilenetv3.yaml')
    parser.add_argument('--ckpt', type=str, default='checkpoints/cnn/best_model.pth')
    args = parser.parse_args()

    base_cfg = load_config(args.base_cfg)
    cnn_cfg = load_config(args.cnn_cfg)
    
    set_seed(base_cfg.project.seed)
    device = get_device(base_cfg.project.device)
    
    results_dir = Path(base_cfg.paths.results) / "stage3"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Plot training curves if history exists
    history_path = Path(base_cfg.paths.checkpoints) / "cnn" / "history.json"
    plot_training_curves(history_path, results_dir)
    
    # Load Model
    model = RiskModel(cnn_cfg).to(device)
    ckpt_path = Path(args.ckpt)
    
    if not ckpt_path.exists():
        print(f"Checkpoint not found at {ckpt_path}. Cannot perform evaluation.")
        print("Please train the model first using 'python scripts/train_cnn.py'")
        # Create a dummy json for progression
        with open(results_dir / "metrics.json", "w") as f:
            json.dump({"error": "Model not trained yet."}, f)
        return
        
    print(f"Loading checkpoint from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Data Loaders
    print("Loading datasets...")
    ai4mars_cfg = load_config("configs/datasets/ai4mars.yaml")
    hirise_cfg = load_config("configs/datasets/hirise.yaml")
    ctx_cfg = load_config("configs/datasets/ctx.yaml")
    
    test_ds = AI4MarsDataset.from_config(base_cfg, ai4mars_cfg, split="test")
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False)
    
    hirise_ds = HiRISEDataset.from_config(base_cfg, hirise_cfg)
    hirise_loader = DataLoader(hirise_ds, batch_size=4, shuffle=False)
    
    ctx_ds = CTXDataset.from_config(base_cfg, ctx_cfg, max_tiles=100)
    ctx_loader = DataLoader(ctx_ds, batch_size=1, shuffle=False)
    
    # 2. Evaluate AI4Mars Test Set
    print("\nEvaluating on AI4Mars Test Set...")
    ai4mars_metrics = evaluate_ai4mars(model, test_loader, device, results_dir)
    print("AI4Mars Metrics:")
    for k, v in ai4mars_metrics.items():
        print(f"  {k}: {v:.4f}")
        
    # 3. Evaluate HiRISE Cross-domain
    print("\nEvaluating on HiRISE Cross-domain...")
    hirise_metrics = evaluate_hirise(model, hirise_loader, device)
    print("HiRISE Metrics:")
    for k, v in hirise_metrics.items():
        print(f"  {k}: {v:.4f}")
        
    # 4. Evaluate CTX
    print("\nGenerating CTX qualitative heatmaps...")
    evaluate_ctx(model, ctx_loader, device, results_dir, base_cfg)
    
    # Save all metrics
    metrics_out = {
        'ai4mars': ai4mars_metrics,
        'hirise': hirise_metrics
    }
    with open(results_dir / "metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)
    print(f"\nSaved metrics to {results_dir / 'metrics.json'}")

if __name__ == "__main__":
    main()
