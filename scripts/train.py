#!/usr/bin/env python3
"""
Badger training script.

Usage:
    # Train from scratch
    python scripts/train.py --config config/default.yaml

    # Resume from checkpoint
    python scripts/train.py --config config/default.yaml --resume runs/exp/checkpoint.pth

    # Quick test on a small subset
    python scripts/train.py --config config/default.yaml --debug

    # Run an experiment combo
    python scripts/train.py --config config/default.yaml --experiment simota_assigner,ema_weights
"""

import os
import sys
import argparse
import yaml
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models import Badger, create_model
from src.losses import BadgerLoss
from src.data import create_dataloader
from src.utils import Logger, save_checkpoint, load_checkpoint, setup_experiment_dir
from src.utils.benchmark import quick_benchmark
from experiments import EXPERIMENTS


def parse_args():
    parser = argparse.ArgumentParser(description='Train Badger object detector')
    parser.add_argument('--config', type=str, default='config/default.yaml',
                        help='Path to config YAML')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    parser.add_argument('--experiment', type=str, default=None,
                        help='Comma-separated experiment names to enable')
    parser.add_argument('--debug', action='store_true',
                        help='Quick debug mode (small data, few epochs)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device: cuda or cpu')
    return parser.parse_args()


def load_config(config_path):
    """Load YAML configuration."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def train_one_epoch(model, dataloader, loss_fn, optimizer, scaler, device, epoch, logger):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')

    for batch_idx, (images, targets) in enumerate(pbar):
        images = images.to(device)
        targets = targets.to(device)

        # Mixed precision forward pass
        with torch.autocast(device_type='cuda' if 'cuda' in device else 'cpu'):
            cls_scores, bbox_preds = model(images)
            img_size = images.shape[-2:]
            loss, loss_dict = loss_fn(cls_scores, bbox_preds, targets, img_size)

        # Backward pass
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.3f}',
            'box': f'{loss_dict["box"]:.3f}',
            'cls': f'{loss_dict["cls"]:.3f}',
        })

    avg_loss = total_loss / len(dataloader)
    return avg_loss


@torch.no_grad()
def validate(model, dataloader, loss_fn, device):
    """Validate the model."""
    model.eval()
    total_loss = 0.0

    pbar = tqdm(dataloader, desc='Validating')
    for images, targets in pbar:
        images = images.to(device)
        targets = targets.to(device)

        cls_scores, bbox_preds = model(images)
        img_size = images.shape[-2:]
        loss, loss_dict = loss_fn(cls_scores, bbox_preds, targets, img_size)

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss


def main():
    args = parse_args()

    # Load config
    config = load_config(args.config)
    cfg_model = config['model']
    cfg_train = config['training']
    cfg_data = config['data']

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Debug mode overrides
    if args.debug:
        cfg_train['epochs'] = 5
        cfg_train['batch_size'] = 2
        print("DEBUG MODE: reduced epochs and batch size")

    # Create experiment directory
    exp_name = args.experiment.replace(',', '_') if args.experiment else 'baseline'
    exp_dir = setup_experiment_dir(base_dir=config['logging']['log_dir'],
                                    experiment_name=exp_name)
    print(f"Experiment directory: {exp_dir}")

    # Save config for reproducibility
    with open(exp_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)

    # Create model
    print("\nCreating model...")
    model = create_model(
        variant='badger-s',
        num_classes=cfg_model['head']['num_classes']
    )
    model = model.to(device)

    total_params, trainable_params = model.count_parameters()
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Quick benchmark
    print("\nQuick benchmark before training:")
    quick_benchmark(model, device=str(device))

    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader = create_dataloader(
        dataset_yaml=cfg_data['dataset_yaml'],
        img_size=cfg_train['img_size'],
        batch_size=cfg_train['batch_size'],
        augment=True,
        mosaic=True,
        num_workers=cfg_data['num_workers'],
        shuffle=True
    )

    # Loss function
    assigner = cfg_train.get('loss', {}).get('assigner', 'tal')
    loss_fn = BadgerLoss(
        num_classes=cfg_model['head']['num_classes'],
        box_weight=cfg_train['loss']['box_weight'],
        cls_weight=cfg_train['loss']['cls_weight'],
        dfl_weight=cfg_train['loss']['dfl_weight'],
        label_smoothing=cfg_train.get('label_smoothing', 0.0),
        assigner=assigner
    )
    print(f"  Loss assigner: {assigner}")

    # Apply experiments
    if args.experiment:
        experiment_names = [e.strip() for e in args.experiment.split(',')]
        print(f"\nApplying experiments: {experiment_names}")
        for name in experiment_names:
            if name not in EXPERIMENTS:
                print(f"  Warning: unknown experiment '{name}', skipping")
                continue
            exp = EXPERIMENTS[name]
            print(f"  ✅ {exp['name']}: {exp['description']}")
            print(f"     Expected gain: {exp['expected_gain']}")

    # Optimizer
    if cfg_train['optimizer'] == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg_train['lr'],
            weight_decay=cfg_train['weight_decay']
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=cfg_train['lr'],
            momentum=cfg_train['momentum'],
            weight_decay=cfg_train['weight_decay']
        )

    # Mixed precision scaler
    scaler = torch.amp.GradScaler('cuda') if 'cuda' in str(device) else None

    # Learning rate scheduler
    if cfg_train['lr_scheduler'] == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg_train['epochs'],
            eta_min=cfg_train['lr'] * cfg_train['lr_final']
        )
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=cfg_train['epochs'] // 3,
            gamma=0.1
        )

    # Resume if requested
    start_epoch = 0
    if args.resume:
        print(f"\nResuming from {args.resume}")
        start_epoch, _, _ = load_checkpoint(args.resume, model, optimizer)

    # Logger
    logger = Logger(
        log_dir=str(exp_dir),
        use_tensorboard=config['logging'].get('tensorboard', True),
        use_wandb=config['logging'].get('wandb', False)
    )

    # Training loop
    print(f"\n{'='*60}")
    print(f"  Starting training — {cfg_train['epochs']} epochs")
    print(f"{'='*60}\n")

    best_loss = float('inf')

    for epoch in range(start_epoch, cfg_train['epochs']):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, loss_fn, optimizer, scaler,
            device, epoch, logger
        )

        # Log
        current_lr = optimizer.param_groups[0]['lr']
        logger.log_scalars({
            'train/loss': train_loss,
            'train/lr': current_lr,
            'train/epoch': epoch,
        }, epoch)

        print(f"\n  Epoch {epoch+1}/{cfg_train['epochs']} — "
              f"Train Loss: {train_loss:.4f}, LR: {current_lr:.6f}")

        # Save checkpoint
        if (epoch + 1) % config['logging']['save_interval'] == 0:
            ckpt_path = exp_dir / f'checkpoint_epoch_{epoch+1}.pth'
            save_checkpoint(model, optimizer, epoch, train_loss, str(ckpt_path), config)
            print(f"  Saved checkpoint: {ckpt_path}")

        # Save best
        if train_loss < best_loss:
            best_loss = train_loss
            best_path = exp_dir / 'best.pth'
            save_checkpoint(model, optimizer, epoch, train_loss, str(best_path), config)

        # Update LR
        scheduler.step()

    # Final benchmark
    print(f"\n{'='*60}")
    print(f"  Training complete!")
    print(f"{'='*60}")

    print("\nFinal benchmark:")
    quick_benchmark(model, device=str(device))

    # Save final model
    final_path = exp_dir / 'final.pth'
    save_checkpoint(model, optimizer, cfg_train['epochs'] - 1, train_loss,
                    str(final_path), config)
    print(f"\nFinal model saved to: {final_path}")

    logger.close()
    print("Done!")


if __name__ == '__main__':
    main()
