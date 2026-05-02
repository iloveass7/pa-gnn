import sys
import argparse
from pathlib import Path
import time
from collections import defaultdict
import json

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

# Add project root
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import load_config
from src.utils.seed import set_seed, get_device
from src.utils.logger import get_logger, setup_logger
from src.data.loaders.ai4mars_loader import AI4MarsDataset
from src.models.fusion.fusion_model import EndToEndFusionModel
from src.models.gnn.graph_builder import GraphBuilder
from src.models.gnn.gatv2 import PAGATv2
from src.training.weak_labels import compute_weak_labels

def evaluate_gnn(model, dataloader, fusion_model, graph_builder, device, gat_cfg):
    model.eval()
    
    all_targets = []
    all_preds = []
    total_loss = 0.0
    
    num_batches = len(dataloader)
    
    for images, targets, _ in dataloader:
        images = images.to(device)
        targets = targets.to(device)
        
        # We use batch size 1 for simplicity in GNN training here
        # Or if batched, we need PyG DataLoader. For now, assuming batch=1
        for i in range(images.size(0)):
            img = images[i:i+1]
            tgt = targets[i:i+1]
            
            with torch.no_grad():
                fusion_dict = fusion_model(img)
                data = graph_builder.build(img, fusion_dict, target=tgt)
                
                if data.x.size(0) == 0: continue
                
                # Forward pass
                preds = model(data.x, data.edge_index, data.edge_attr)
                
                # Active and valid mask
                valid_mask = (data.y >= 0) & data.active_mask
                if valid_mask.sum() == 0: continue
                
                valid_preds = preds[valid_mask]
                valid_targets = data.y[valid_mask]
                
                # Binarize targets for metric
                hard_targets = (valid_targets > 0.5).float()
                
                all_targets.extend(hard_targets.cpu().numpy())
                all_preds.extend(valid_preds.cpu().numpy())
                
                # Loss
                weight = torch.ones_like(hard_targets)
                weight[hard_targets > 0.5] = gat_cfg.training.loss.positive_weight
                loss = F.binary_cross_entropy(valid_preds, hard_targets, weight=weight)
                total_loss += loss.item()
                
    total_loss /= num_batches if num_batches > 0 else 1
    
    metrics = {'val_loss': total_loss}
    if len(all_targets) > 0:
        try:
            metrics['val_auc_roc'] = roc_auc_score(all_targets, all_preds)
        except ValueError:
            metrics['val_auc_roc'] = 0.5 # Only one class present
            
        # Hazard Recall
        t = np.array(all_targets)
        p = np.array(all_preds) > 0.5
        tp = ((p == 1) & (t == 1)).sum()
        fn = ((p == 0) & (t == 1)).sum()
        metrics['val_hazard_recall'] = tp / (tp + fn + 1e-8)
        
    return metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_cfg', type=str, default='configs/base.yaml')
    parser.add_argument('--cnn_cfg', type=str, default='configs/cnn/mobilenetv3.yaml')
    parser.add_argument('--phys_cfg', type=str, default='configs/physics.yaml')
    parser.add_argument('--fusion_cfg', type=str, default='configs/fusion/adaptive_fusion.yaml')
    parser.add_argument('--gat_cfg', type=str, default='configs/gnn/gatv2.yaml')
    args = parser.parse_args()

    base_cfg = load_config(args.base_cfg)
    cnn_cfg = load_config(args.cnn_cfg)
    phys_cfg = load_config(args.phys_cfg)
    fusion_cfg = load_config(args.fusion_cfg)
    gat_cfg = load_config(args.gat_cfg)
    
    set_seed(base_cfg.project.seed)
    device = get_device(base_cfg.project.device)
    
    # Setup directories
    checkpoint_dir = Path(base_cfg.paths.checkpoints) / "gnn"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(base_cfg.paths.logs) / "gnn"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    setup_logger(log_file=log_dir / "train_gnn.log")
    logger = get_logger("TrainGNN")
    logger.info(f"Starting Stage 6 GNN Training on {device}")

    # Models
    fusion_model = EndToEndFusionModel(cnn_cfg, phys_cfg, fusion_cfg, freeze_cnn=True).to(device)
    graph_builder = GraphBuilder(gat_cfg)
    
    gat_model = PAGATv2(gat_cfg).to(device)
    
    # Optimizer
    optimizer = optim.Adam(
        gat_model.parameters(), 
        lr=gat_cfg.training.learning_rate, 
        weight_decay=gat_cfg.training.weight_decay
    )
    
    # Data Loaders (using batch_size=1 since we process graphs individually)
    # For a real pipeline, we'd precompute PyG Data objects and use PyG DataLoader.
    logger.info("Loading datasets (batch_size=1 for on-the-fly graph building)...")
    ai4mars_cfg = load_config("configs/datasets/ai4mars.yaml")
    train_ds = AI4MarsDataset.from_config(base_cfg, ai4mars_cfg, split="train")
    val_ds = AI4MarsDataset.from_config(base_cfg, ai4mars_cfg, split="val")
    
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4)
    
    # Training Loop
    epochs = gat_cfg.training.epochs
    best_auc = 0.0
    patience = gat_cfg.training.early_stopping.patience
    patience_counter = 0
    
    history = defaultdict(list)
    
    for epoch in range(1, epochs + 1):
        gat_model.train()
        total_loss = 0.0
        
        t0 = time.time()
        for batch_idx, (images, targets, _) in enumerate(train_loader):
            images, targets = images.to(device), targets.to(device)
            
            # 1. Build Graph
            with torch.no_grad():
                fusion_dict = fusion_model(images)
                fusion_dict = {k: v.squeeze(0) if isinstance(v, torch.Tensor) else v 
                for k, v in fusion_dict.items()}

                data = graph_builder.build(images[0], fusion_dict, target=targets[0])
            if data.x.size(0) == 0: continue
            
            # 2. Weak Labeling
            active_mask = data.active_mask
            node_labels = data.y
            
            if gat_cfg.training.weak_labeling.enabled:
                # Binarize GT for weak labeling logic (1.0 for hazard > 0.5)
                hard_labels = (node_labels > 0.5).float()
                # Restore ignore regions (-1)
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
                
            # 3. Forward & Loss
            optimizer.zero_grad()
            preds = gat_model(data.x, data.edge_index, data.edge_attr)
            
            valid_mask = (updated_labels >= 0) & active_mask
            if valid_mask.sum() == 0: continue
            
            valid_preds = preds[valid_mask]
            valid_targets = updated_labels[valid_mask]
            
            weight = torch.ones_like(valid_targets)
            weight[valid_targets >= 0.9] = gat_cfg.training.loss.positive_weight # Full hazard
            weight[(valid_targets > 0.5) & (valid_targets < 0.9)] = 1.0 # Weak hazard
            
            loss = F.binary_cross_entropy(valid_preds, valid_targets, weight=weight)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                logger.info(f"Epoch {epoch} [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}")
                
        # Epoch summary
        train_loss = total_loss / len(train_loader)
        history['train_loss'].append(train_loss)
        
        val_metrics = evaluate_gnn(gat_model, val_loader, fusion_model, graph_builder, device, gat_cfg)
        for k, v in val_metrics.items():
            history[k].append(v)
            
        time_elapsed = time.time() - t0
        logger.info(f"Epoch {epoch} Summary ({time_elapsed:.1f}s): Train Loss: {train_loss:.4f}, " + 
                    ", ".join([f"{k}: {v:.4f}" for k, v in val_metrics.items()]))
                    
        # Checkpoint
        current_auc = val_metrics.get('val_auc_roc', 0)
        if current_auc > best_auc:
            best_auc = current_auc
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': gat_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': val_metrics
            }, checkpoint_dir / 'best_gat_model.pth')
            logger.info(f"  New best model saved! AUC: {best_auc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered.")
                break
                
        with open(checkpoint_dir / "history.json", 'w') as f:
            json.dump(history, f, indent=2)

if __name__ == "__main__":
    main()
