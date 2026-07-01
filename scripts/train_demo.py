#!/usr/bin/env python3
"""
Quick training demo — shows the upgraded SuperMind v2 in action.

Runs 30 epochs on synthetic data with:
  - WIoU v3 loss (dynamic non-monotonic focusing)
  - OneCycle LR schedule
  - EMA weight averaging
  - Label smoothing
  - Gradient clipping
  - Mixed precision (if GPU available)

Usage:
    python scripts/train_demo.py
    python scripts/train_demo.py --preset super --epochs 50
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader

from src.models import create_model
from src.training.supermind import SuperMind


def parse_args():
    p = argparse.ArgumentParser(description='Badger SuperMind Training Demo')
    p.add_argument('--preset', default='accurate', choices=['fast','accurate','super'])
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--model', default='badger-n')
    p.add_argument('--device', default='cuda')
    p.add_argument('--num-classes', type=int, default=10)
    p.add_argument('--img-size', type=int, default=320)
    p.add_argument('--num-samples', type=int, default=500)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # ── Synthetic data ──────────────────────────────────────────
    from scripts.test_learning import SyntheticShapesDataset, collate_fn

    print(f"\n{'='*60}")
    print(f"  🦡 Badger SuperMind v2 — Training Demo")
    print(f"{'='*60}")
    print(f"  Preset: {args.preset}")
    print(f"  Model: {args.model}")
    print(f"  Device: {device}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Samples: {args.num_samples} (train) / 100 (val)")

    train_ds = SyntheticShapesDataset(
        num_samples=args.num_samples, img_size=args.img_size,
        num_classes=args.num_classes, seed=42
    )
    val_ds = SyntheticShapesDataset(
        num_samples=100, img_size=args.img_size,
        num_classes=args.num_classes, seed=123
    )

    train_loader = DataLoader(
        train_ds, batch_size=8, shuffle=True,
        collate_fn=collate_fn, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=8, shuffle=False,
        collate_fn=collate_fn, num_workers=0
    )

    # ── Model ───────────────────────────────────────────────────
    model = create_model(variant=args.model, num_classes=args.num_classes)
    total, trainable = model.count_parameters()
    print(f"  Parameters: {total:,} total")

    # ── Train with SuperMind ────────────────────────────────────
    trainer = SuperMind(
        model, train_loader, val_loader=val_loader,
        device=str(device),
        project_dir=f'runs/demo_{args.preset}',
        use_amp=('cuda' in str(device)),
        use_ema=True,
        use_compile=('cuda' in str(device)),
    )

    history = trainer.fit(
        epochs=args.epochs,
        num_classes=args.num_classes,
        preset=args.preset,
    )

    # ── Analysis ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  📊 Training Analysis")
    print(f"{'='*60}")

    losses = history['train_loss']
    val_losses = history.get('val_loss', [])
    pct = 0.0

    if len(losses) >= 2:
        first_loss = losses[0]
        last_loss = losses[-1]
        pct = (first_loss - last_loss) / first_loss * 100
        print(f"  Train loss: {first_loss:.4f} → {last_loss:.4f} (-{pct:.1f}%)")
    elif losses:
        print(f"  Train loss: {losses[-1]:.4f}")

    if val_losses:
        first_v = val_losses[0]
        last_v = val_losses[-1]
        pct_v = (first_v - last_v) / first_v * 100
        print(f"  Val loss:   {first_v:.4f} → {last_v:.4f} (-{pct_v:.1f}%)")
        gaps = history.get('generalization_gap', [])
        actions = [a for a in history.get('stability_action', []) if a]
        if gaps:
            print(f"  Gap:        {gaps[0]:.4f} → {gaps[-1]:.4f}")
        if actions:
            print(f"  Stability: {len(actions)} LR adjustment(s)")

    # Component analysis
    box_losses = history['box']
    cls_losses = history['cls']
    dfl_losses = history['dfl']

    print(f"  Box loss:   {box_losses[0]:.4f} → {box_losses[-1]:.4f} (-{(box_losses[0]-box_losses[-1])/box_losses[0]*100:.1f}%)")
    print(f"  Cls loss:   {cls_losses[0]:.4f} → {cls_losses[-1]:.4f} (-{(cls_losses[0]-cls_losses[-1])/max(cls_losses[0],1e-8)*100:.1f}%)")
    print(f"  DFL loss:   {dfl_losses[0]:.4f} → {dfl_losses[-1]:.4f} (-{(dfl_losses[0]-dfl_losses[-1])/max(dfl_losses[0],1e-8)*100:.1f}%)")

    # Check if training was healthy
    decreasing = sum(1 for i in range(1, len(losses)) if losses[i] < losses[i-1])
    trend = decreasing / max(1, len(losses) - 1) * 100

    if pct > 20:
        verdict = "✅ EXCELLENT — Model learning strongly!"
    elif pct > 10:
        verdict = "✅ GOOD — Consistent learning confirmed"
    elif pct > 5:
        verdict = "⚠️  MODERATE — Try more epochs or a larger model"
    else:
        verdict = "❌ POOR — Check data or hyperparameters"

    print(f"\n  Loss trend: {decreasing}/{len(losses)-1} decreasing ({trend:.0f}%)")
    print(f"  {verdict}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
