import sys
import argparse
from pathlib import Path

# Add project root
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from src.utils.config import load_config
from src.utils.seed import set_seed, get_device
from src.utils.logger import get_logger, setup_logger
from src.data.loaders.ai4mars_loader import AI4MarsDataset
from src.models.fusion.fusion_model import EndToEndFusionModel
from src.training.losses import RiskLoss
from src.training.trainer import Trainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_cfg', type=str, default='configs/base.yaml')
    parser.add_argument('--cnn_cfg', type=str, default='configs/cnn/mobilenetv3.yaml')
    parser.add_argument('--phys_cfg', type=str, default='configs/physics.yaml')
    parser.add_argument('--fusion_cfg', type=str, default='configs/fusion/adaptive_fusion.yaml')
    parser.add_argument('--ds_cfg', type=str, default='configs/datasets/ai4mars.yaml')
    parser.add_argument('--cnn_ckpt', type=str, default=None, help="Path to trained CNN checkpoint")
    args = parser.parse_args()

    base_cfg = load_config(args.base_cfg)
    cnn_cfg = load_config(args.cnn_cfg)
    phys_cfg = load_config(args.phys_cfg)
    fusion_cfg = load_config(args.fusion_cfg)
    ds_cfg = load_config(args.ds_cfg)

    set_seed(base_cfg.project.seed)
    device = get_device(base_cfg.project.device)
    
    # Setup directories
    checkpoint_dir = Path(base_cfg.paths.checkpoints) / "fusion"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(base_cfg.paths.logs) / "fusion"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    setup_logger(log_file=log_dir / "train_fusion.log")
    logger = get_logger("TrainFusion")
    logger.info(f"Starting Stage 4 Fusion Training on {device}")

    # Data
    logger.info("Loading datasets...")
    train_ds = AI4MarsDataset.from_config(base_cfg, ds_cfg, split="train")
    val_ds = AI4MarsDataset.from_config(base_cfg, ds_cfg, split="val")
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=cnn_cfg.training.batch_size, 
        shuffle=ds_cfg.dataloader.shuffle_train if hasattr(ds_cfg.dataloader, 'shuffle_train') else True,
        num_workers=ds_cfg.dataloader.num_workers,
        pin_memory=ds_cfg.dataloader.pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_ds, 
        batch_size=cnn_cfg.training.batch_size, 
        shuffle=False,
        num_workers=ds_cfg.dataloader.num_workers,
        pin_memory=ds_cfg.dataloader.pin_memory
    )
    
    # Model
    logger.info("Initializing Fusion model...")
    freeze_cnn = not fusion_cfg.training.joint_with_cnn
    logger.info(f"CNN frozen: {freeze_cnn}")
    
    model = EndToEndFusionModel(cnn_cfg, phys_cfg, fusion_cfg, freeze_cnn=freeze_cnn)
    
    if args.cnn_ckpt:
        logger.info(f"Loading CNN checkpoint from {args.cnn_ckpt}")
        ckpt = torch.load(args.cnn_ckpt, map_location='cpu', weights_only=True)
        model.cnn.load_state_dict(ckpt['model_state_dict'])
        
    model = model.to(device)
    
    # Loss
    criterion = RiskLoss(
        bce_weight=cnn_cfg.training.loss.bce_weight,
        bce_hazard_weight=cnn_cfg.training.loss.bce_hazard_weight,
        dice_weight=cnn_cfg.training.loss.dice_weight,
        dice_threshold=cnn_cfg.training.loss.dice_threshold,
        tv_weight=cnn_cfg.training.loss.tv_weight
    )
    
    # Optimizer (only optimize parameters that require grad)
    params_to_opt = [p for p in model.parameters() if p.requires_grad]
    logger.info(f"Optimizing {sum(p.numel() for p in params_to_opt)} parameters")
    
    optimizer = optim.AdamW(
        params_to_opt, 
        lr=fusion_cfg.training.learning_rate, 
        weight_decay=fusion_cfg.training.weight_decay
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=cnn_cfg.training.scheduler_T_max
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
        config=cnn_cfg, # early stopping uses cnn settings
        save_dir=checkpoint_dir
    )
    
    # Train
    trainer.fit(train_loader, val_loader, num_epochs=cnn_cfg.training.epochs)
    logger.info("Training complete.")

if __name__ == "__main__":
    main()
