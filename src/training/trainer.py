import time
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict
import json

from src.evaluation.metrics import compute_metrics
from src.utils.logger import get_logger

logger = get_logger("Trainer")

class Trainer:
    def __init__(self, model, optimizer, scheduler, criterion, device, config, save_dir):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.config = config
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.best_metric = -float('inf')
        if config.training.early_stopping.mode == 'min':
            self.best_metric = float('inf')
            
        self.patience = config.training.early_stopping.patience
        self.patience_counter = 0
        self.monitor_metric = config.training.early_stopping.monitor
        self.mode = config.training.early_stopping.mode
        
        self.history = defaultdict(list)

    def train_epoch(self, dataloader, epoch):
        self.model.train()
        epoch_losses = defaultdict(float)
        
        t0 = time.time()
        for batch_idx, (images, targets, _) in enumerate(dataloader):
            images, targets = images.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            preds = self.model(images)
            loss_dict = self.criterion(preds, targets)
            
            loss = loss_dict['loss']
            loss.backward()
            self.optimizer.step()
            
            for k, v in loss_dict.items():
                if isinstance(v, torch.Tensor):
                    epoch_losses[k] += v.item()
                else:
                    epoch_losses[k] += v
                
            if batch_idx % 50 == 0:
                logger.info(f"Epoch {epoch} [{batch_idx}/{len(dataloader)}] Loss: {loss.item():.4f}")
                
        # Average losses
        for k in epoch_losses.keys():
            epoch_losses[k] /= len(dataloader)
            self.history[f"train_{k}"].append(epoch_losses[k])
            
        time_elapsed = time.time() - t0
        logger.info(f"Epoch {epoch} Train Summary ({time_elapsed:.1f}s): " + 
                    ", ".join([f"{k}: {v:.4f}" for k, v in epoch_losses.items()]))
        
        return epoch_losses

    @torch.no_grad()
    def evaluate(self, dataloader, epoch, prefix="val"):
        self.model.eval()
        epoch_losses = defaultdict(float)
        all_metrics = defaultdict(float)
        
        t0 = time.time()
        for images, targets, _ in dataloader:
            images, targets = images.to(self.device), targets.to(self.device)
            
            preds = self.model(images)
            loss_dict = self.criterion(preds, targets)
            
            for k, v in loss_dict.items():
                if isinstance(v, torch.Tensor):
                    epoch_losses[k] += v.item()
                else:
                    epoch_losses[k] += v
                    
            # Compute metrics
            batch_metrics = compute_metrics(preds, targets)
            for k, v in batch_metrics.items():
                all_metrics[k] += v
                
        # Average
        num_batches = len(dataloader)
        if num_batches == 0: return epoch_losses, all_metrics
        for k in epoch_losses.keys():
            epoch_losses[k] /= num_batches
            self.history[f"{prefix}_{k}"].append(epoch_losses[k])
            
        for k in all_metrics.keys():
            all_metrics[k] /= num_batches
            self.history[f"{prefix}_{k}"].append(all_metrics[k])
            
        time_elapsed = time.time() - t0
        logger.info(f"Epoch {epoch} {prefix.capitalize()} Summary ({time_elapsed:.1f}s): " + 
                    ", ".join([f"{k}: {v:.4f}" for k, v in epoch_losses.items()]) + 
                    " | " + 
                    ", ".join([f"{k}: {v:.4f}" for k, v in all_metrics.items()]))
                    
        return epoch_losses, all_metrics

    def fit(self, train_loader, val_loader, num_epochs):
        logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(1, num_epochs + 1):
            self.train_epoch(train_loader, epoch)
            _, val_metrics = self.evaluate(val_loader, epoch)
            
            if self.scheduler is not None:
                self.scheduler.step()
                
            # Early stopping and checkpointing
            current_metric = val_metrics[self.monitor_metric.replace('val_', '')]
            
            is_best = False
            if self.mode == 'max' and current_metric > self.best_metric:
                self.best_metric = current_metric
                is_best = True
            elif self.mode == 'min' and current_metric < self.best_metric:
                self.best_metric = current_metric
                is_best = True
                
            if is_best:
                self.patience_counter = 0
                self.save_checkpoint('best_model.pth', epoch, val_metrics)
                logger.info(f"  New best model saved! {self.monitor_metric}: {self.best_metric:.4f}")
            else:
                self.patience_counter += 1
                logger.info(f"  No improvement. Patience: {self.patience_counter}/{self.patience}")
                
            if self.patience_counter >= self.patience:
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break
                
        # Save final history
        self.save_history()
        return self.history

    def save_checkpoint(self, filename, epoch, metrics):
        path = self.save_dir / filename
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics
        }, path)

    def save_history(self):
        with open(self.save_dir / "history.json", 'w') as f:
            json.dump(self.history, f, indent=2)
