#!/usr/bin/env python3
"""
Synthetic learning test — verifies Badger ACTUALLY learns.

Creates a tiny synthetic dataset (colored squares on black background)
and trains Badger-Nano for a few epochs. If the loss doesn't drop,
something is broken.

Usage:
    python scripts/test_learning.py
    python scripts/test_learning.py --model badger-s --epochs 20
"""

import sys
import argparse
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from src.models import create_model
from src.losses import BadgerLoss


# =============================================================================
# Synthetic Dataset — colored squares on black background
# =============================================================================

class SyntheticShapesDataset(Dataset):
    """
    Generates images with simple colored rectangles.
    Perfect for verifying the training loop works.

    Each image: black background (320×320), 1-3 random colored squares.
    Model must learn to detect them.
    """

    COLORS = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
        (255, 128, 0),  # Orange
        (128, 0, 255),  # Purple
        (0, 128, 255),  # Sky blue
        (255, 128, 128), # Pink
    ]

    def __init__(self, num_samples=500, img_size=320, num_classes=10,
                 seed=42):
        self.num_samples = num_samples
        self.img_size = img_size
        self.num_classes = num_classes
        self.rng = np.random.RandomState(seed)
        self._samples = [self._generate() for _ in range(num_samples)]

    def _generate(self):
        """Generate one synthetic image with random rectangles."""
        img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        boxes = []
        classes = []

        # 1-3 objects per image
        num_objs = self.rng.randint(1, 4)

        for _ in range(num_objs):
            cls = self.rng.randint(0, min(self.num_classes, len(self.COLORS)))
            color = self.COLORS[cls]

            # Random size: 20-80 pixels
            w = int(self.rng.uniform(20, 80))
            h = int(self.rng.uniform(20, 80))

            # Random position (ensure it fits)
            x1 = int(self.rng.uniform(0, self.img_size - w - 1))
            y1 = int(self.rng.uniform(0, self.img_size - h - 1))
            x2 = x1 + w
            y2 = y1 + h

            # Draw filled rectangle
            cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)

            # Normalize box: cx, cy, w, h (0-1)
            cx = (x1 + x2) / 2 / self.img_size
            cy = (y1 + y2) / 2 / self.img_size
            nw = w / self.img_size
            nh = h / self.img_size
            boxes.append([cx, cy, nw, nh])
            classes.append(cls)

        boxes = np.array(boxes, dtype=np.float32)
        classes = np.array(classes, dtype=np.int64).reshape(-1, 1)

        # Build target: [num_obj, 6] = [batch_idx, cls, cx, cy, w, h]
        targets = np.concatenate([
            np.zeros((len(boxes), 1), dtype=np.float32),
            classes.astype(np.float32),
            boxes
        ], axis=1)

        return img, targets

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        img, targets = self._samples[idx]

        # Convert to tensor: BGR->RGB, HWC->CHW, normalize
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        targets = torch.from_numpy(targets).float()

        return img, targets


def collate_fn(batch):
    """Collate variable-length targets."""
    images, targets = zip(*batch)
    images = torch.stack(images, 0)

    # Add batch index to targets
    batch_targets = []
    for i, t in enumerate(targets):
        if len(t) > 0:
            t = t.clone()
            t[:, 0] = i
            batch_targets.append(t)

    if batch_targets:
        batch_targets = torch.cat(batch_targets, 0)
    else:
        batch_targets = torch.zeros((0, 6), dtype=torch.float32)

    return images, batch_targets


# =============================================================================
# Training Loop
# =============================================================================

def train_epoch(model, dataloader, loss_fn, optimizer, scaler, device):
    model.train()
    total_loss = 0.0
    total_box = 0.0
    total_cls = 0.0
    total_dfl = 0.0

    pbar = tqdm(dataloader, desc='Train', leave=False)
    for images, targets in pbar:
        images = images.to(device)
        targets = targets.to(device)

        with torch.amp.autocast('cuda' if 'cuda' in str(device) else 'cpu'):
            cls_scores, bbox_preds, raw_reg = model(images, return_raw_reg=True)
            img_size = images.shape[-2:]
            loss, loss_dict = loss_fn(cls_scores, bbox_preds, targets, img_size,
                                       raw_reg_preds=raw_reg)

        optimizer.zero_grad()
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        total_box += loss_dict.get('box', 0)
        total_cls += loss_dict.get('cls', 0)
        total_dfl += loss_dict.get('dfl', 0)

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    n = len(dataloader)
    return total_loss / n, total_box / n, total_cls / n, total_dfl / n


@torch.no_grad()
def validate_epoch(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0.0

    for images, targets in dataloader:
        images = images.to(device)
        targets = targets.to(device)

        cls_scores, bbox_preds, raw_reg = model(images, return_raw_reg=True)
        img_size = images.shape[-2:]
        loss, _ = loss_fn(cls_scores, bbox_preds, targets, img_size,
                          raw_reg_preds=raw_reg)
        total_loss += loss.item()

    return total_loss / len(dataloader)


# =============================================================================
# Main
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description='Synthetic learning test')
    p.add_argument('--model', default='badger-n', help='Model variant')
    p.add_argument('--epochs', type=int, default=15, help='Training epochs')
    p.add_argument('--batch-size', type=int, default=8, help='Batch size')
    p.add_argument('--device', default='cuda', help='Device')
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Create dataset
    print("\n=== Creating Synthetic Dataset ===")
    train_ds = SyntheticShapesDataset(num_samples=500, img_size=320, num_classes=10)
    val_ds = SyntheticShapesDataset(num_samples=100, img_size=320, num_classes=10, seed=123)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, collate_fn=collate_fn, num_workers=0)
    print(f"  Train: {len(train_ds)} images, {len(train_loader)} batches")
    print(f"  Val:   {len(val_ds)} images, {len(val_loader)} batches")

    # Show a sample
    sample_img, sample_targets = train_ds[0]
    print(f"  Sample image shape: {sample_img.shape}")
    print(f"  Sample has {len(sample_targets)} objects")

    # Create model
    print(f"\n=== Creating Model: {args.model} ===")
    model = create_model(variant=args.model, num_classes=10)
    model = model.to(device)
    total, trainable = model.count_parameters()
    print(f"  Parameters: {total:,} total, {trainable:,} trainable")

    # Loss function
    loss_fn = BadgerLoss(
        num_classes=10,
        box_weight=7.5,
        cls_weight=0.5,
        dfl_weight=1.5,
        assigner='tal'
    )

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    scaler = torch.amp.GradScaler('cuda') if 'cuda' in str(device) else None

    # Training
    print(f"\n=== Training {args.epochs} epochs ===")
    history = {'train_loss': [], 'val_loss': [], 'box': [], 'cls': [], 'dfl': []}

    for epoch in range(args.epochs):
        train_loss, box_loss, cls_loss, dfl_loss = train_epoch(
            model, train_loader, loss_fn, optimizer, scaler, device
        )
        scheduler.step()

        val_loss = validate_epoch(model, val_loader, loss_fn, device)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['box'].append(box_loss)
        history['cls'].append(cls_loss)
        history['dfl'].append(dfl_loss)

        lr = optimizer.param_groups[0]['lr']
        print(f"  Epoch {epoch+1:3d}/{args.epochs} | "
              f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
              f"Box: {box_loss:.4f} | Cls: {cls_loss:.4f} | DFL: {dfl_loss:.4f} | "
              f"LR: {lr:.6f}")

    # Analysis
    print(f"\n=== Learning Analysis ===")
    first_loss = history['train_loss'][0]
    last_loss = history['train_loss'][-1]
    delta = first_loss - last_loss
    pct = (delta / first_loss) * 100

    print(f"  Initial loss:  {first_loss:.4f}")
    print(f"  Final loss:    {last_loss:.4f}")
    print(f"  Delta:         {delta:.4f} ({pct:.1f}% reduction)")

    # Check if loss is decreasing consistently
    losses = history['train_loss']
    decreases = sum(1 for i in range(1, len(losses)) if losses[i] < losses[i-1])
    trend_ratio = decreases / max(1, len(losses) - 1)
    print(f"  Decreasing epochs: {decreases}/{len(losses)-1} ({trend_ratio:.0%})")

    # Validate learning
    if pct > 10:
        print(f"\n  ✅ MODEL IS LEARNING — loss dropped {pct:.1f}%")
        passed = True
    elif pct > 5:
        print(f"\n  ⚠️  MARGINAL LEARNING — loss dropped {pct:.1f}% (may need more epochs)")
        passed = True
    else:
        print(f"\n  ❌ MODEL IS NOT LEARNING — loss dropped only {pct:.1f}%")
        passed = False

    # Box loss check (regression)
    box_first = history['box'][0]
    box_last = history['box'][-1]
    print(f"  Box loss: {box_first:.4f} → {box_last:.4f} ({(box_first-box_last)/box_first*100:.1f}% drop)")

    # Cls loss check
    cls_first = history['cls'][0]
    cls_last = history['cls'][-1]
    print(f"  Cls loss: {cls_first:.4f} → {cls_last:.4f} ({(cls_first-cls_last)/max(cls_first,1e-8)*100:.1f}% drop)")

    print(f"\n{'='*60}")
    if passed:
        print("  RESULT: Training verified — Badger learns correctly!")
    else:
        print("  RESULT: Training FAILED — model did not learn.")
    print(f"{'='*60}")

    return 0 if passed else 1


if __name__ == '__main__':
    sys.exit(main())
