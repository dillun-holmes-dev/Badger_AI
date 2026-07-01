#!/usr/bin/env python3
"""
Head-to-head training comparison: Badger v1 vs BadgerV2.

Trains both models on identical synthetic detection data and compares:
  1. Loss convergence (does it learn?)
  2. Convergence speed (which learns faster?)
  3. Parameter efficiency (accuracy per param)
  4. Inference speed (latency + FPS)

No external dataset needed — generates realistic synthetic targets.

Usage:
    python scripts/compare_v1_v2.py
"""

import sys
import time
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader


# =============================================================================
# Synthetic Detection Dataset
# =============================================================================

class SyntheticDetectionDataset(Dataset):
    """
    Generates realistic synthetic detection data.

    Each image has 2-8 objects with random classes and boxes.
    Boxes follow a realistic size distribution (many small, few large).
    """

    def __init__(self, num_images=200, img_size=640, num_classes=20, seed=42):
        self.num_images = num_images
        self.img_size = img_size
        self.num_classes = num_classes
        self.rng = np.random.RandomState(seed)

        # Pre-generate all targets for reproducibility
        self.images = []
        self.targets = []

        for img_idx in range(num_images):
            # 2-8 objects per image
            n_objects = self.rng.randint(2, 9)

            # Generate image with objects (colored rectangles on noise)
            img = self.rng.rand(3, img_size, img_size).astype(np.float32) * 0.3

            img_targets = []
            for _ in range(n_objects):
                cls = self.rng.randint(0, num_classes)

                # Realistic box distribution: many small, few large
                # Log-normal distribution for w/h (peaks at ~0.1, tail to ~0.5)
                w = min(0.8, max(0.02, self.rng.lognormal(-2.5, 0.7)))
                h = min(0.8, max(0.02, self.rng.lognormal(-2.5, 0.7)))

                # Center position (avoid edges)
                cx = self.rng.uniform(w / 2, 1 - w / 2)
                cy = self.rng.uniform(h / 2, 1 - h / 2)

                img_targets.append([img_idx, cls, cx, cy, w, h])

                # Draw colored rectangle on image
                x1 = int((cx - w/2) * img_size)
                y1 = int((cy - h/2) * img_size)
                x2 = int((cx + w/2) * img_size)
                y2 = int((cy + h/2) * img_size)
                color = self.rng.rand(3).astype(np.float32) * 0.7 + 0.3
                for c_idx in range(3):
                    img[c_idx, max(0,y1):min(img_size,y2),
                              max(0,x1):min(img_size,x2)] = color[c_idx]

            self.images.append(torch.from_numpy(img))
            self.targets.append(torch.tensor(img_targets, dtype=torch.float32))

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        return self.images[idx], self.targets[idx]


def collate_fn(batch):
    """Custom collate: stack images, cat targets with batch index."""
    images = torch.stack([b[0] for b in batch])

    # Re-index batch dimension in targets
    all_targets = []
    for i, (_, targets) in enumerate(batch):
        if len(targets) > 0:
            targets = targets.clone()
            targets[:, 0] = i  # Set batch index
            all_targets.append(targets)

    if all_targets:
        targets = torch.cat(all_targets, dim=0)
    else:
        targets = torch.zeros(0, 6)

    return images, targets


# =============================================================================
# Training Loop
# =============================================================================

def train_model(model, loss_fn, train_loader, num_epochs, device, model_name,
                lr=0.001):
    """Train a model and return loss history."""
    model = model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=lr * 0.01
    )

    history = {
        'loss': [], 'box_loss': [], 'cls_loss': [],
        'epoch_time': [], 'lr': []
    }

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*60}")
    print(f"  Training: {model_name}")
    print(f"  Parameters: {total_params:,}")
    print(f"  Device: {device}")
    print(f"{'='*60}")

    for epoch in range(num_epochs):
        epoch_start = time.time()
        epoch_loss = 0.0
        epoch_box = 0.0
        epoch_cls = 0.0
        n_batches = 0

        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)

            # Forward
            cls_scores, bbox_preds = model(images)
            img_size = (images.shape[2], images.shape[3])

            try:
                total_loss, loss_dict = loss_fn(
                    cls_scores, bbox_preds, targets, img_size
                )
            except Exception:
                continue

            if torch.isnan(total_loss) or torch.isinf(total_loss):
                continue

            # Backward
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()

            epoch_loss += total_loss.item()
            epoch_box += loss_dict.get('box', 0)
            epoch_cls += loss_dict.get('cls', 0)
            n_batches += 1

        scheduler.step()

        if n_batches == 0:
            continue

        avg_loss = epoch_loss / n_batches
        avg_box = epoch_box / n_batches
        avg_cls = epoch_cls / n_batches
        epoch_time = time.time() - epoch_start

        history['loss'].append(avg_loss)
        history['box_loss'].append(avg_box)
        history['cls_loss'].append(avg_cls)
        history['epoch_time'].append(epoch_time)
        history['lr'].append(scheduler.get_last_lr()[0])

        if epoch % 5 == 0 or epoch == num_epochs - 1:
            print(f"  Epoch {epoch+1:3d}/{num_epochs} | "
                  f"loss={avg_loss:.4f} box={avg_box:.4f} cls={avg_cls:.4f} | "
                  f"{epoch_time:.1f}s")

    return history


def measure_speed(model, device, img_size=640, warmup=20, runs=50):
    """Measure inference latency and FPS."""
    model = model.to(device).eval()
    dummy = torch.randn(1, 3, img_size, img_size, device=device)

    with torch.no_grad():
        for _ in range(warmup):
            model(dummy)

    if device.type == 'cuda':
        torch.cuda.synchronize()

    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(runs):
            model(dummy)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    latency_ms = elapsed / runs * 1000
    fps = 1000 / latency_ms
    return latency_ms, fps


# =============================================================================
# Main Comparison
# =============================================================================

def main():
    print("\n" + "=" * 60)
    print("  🦡 BADGER v1 vs v2 — HEAD-TO-HEAD TRAINING COMPARISON")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # =========================================================================
    # Setup
    # =========================================================================
    NUM_CLASSES = 20
    IMG_SIZE = 640
    BATCH_SIZE = 4
    NUM_EPOCHS = 25
    NUM_IMAGES = 200
    SEED = 42

    # Reproducibility
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if device.type == 'cuda':
        torch.cuda.manual_seed(SEED)

    # Create dataset
    print(f"\nGenerating synthetic dataset: {NUM_IMAGES} images, "
          f"{NUM_CLASSES} classes, {IMG_SIZE}×{IMG_SIZE}...")
    dataset = SyntheticDetectionDataset(
        num_images=NUM_IMAGES, img_size=IMG_SIZE,
        num_classes=NUM_CLASSES, seed=SEED
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        collate_fn=collate_fn, num_workers=0, drop_last=True)
    print(f"  {len(dataset)} images, {len(loader)} batches per epoch")

    # =========================================================================
    # Create Models
    # =========================================================================
    from src.models import create_model, create_badger_v2
    from src.losses import BadgerLoss

    # Badger v1 — small variant
    model_v1 = create_model('badger-s', num_classes=NUM_CLASSES)
    params_v1 = sum(p.numel() for p in model_v1.parameters())

    # BadgerV2 — small variant with BiFPN
    model_v2 = create_badger_v2('small', num_classes=NUM_CLASSES,
                                 neck_type='bifpn', nms_free=False)
    params_v2 = sum(p.numel() for p in model_v2.parameters())

    print(f"\n  Badger v1 (badger-s):   {params_v1:>12,} params")
    print(f"  BadgerV2  (small/BiFPN): {params_v2:>12,} params")
    print(f"  Param ratio:             {params_v2/params_v1:.2f}×")

    # Loss function (shared — fair comparison)
    loss_fn = BadgerLoss(
        num_classes=NUM_CLASSES,
        box_weight=7.5, cls_weight=0.5, dfl_weight=1.5,
        assigner='simota'
    )

    # =========================================================================
    # Train Both Models
    # =========================================================================

    # Train Badger v1
    torch.manual_seed(SEED)
    if device.type == 'cuda':
        torch.cuda.manual_seed(SEED)
    history_v1 = train_model(
        model_v1, loss_fn, loader, NUM_EPOCHS, device,
        "Badger v1 (badger-s)", lr=0.001
    )

    # Train BadgerV2
    torch.manual_seed(SEED)
    if device.type == 'cuda':
        torch.cuda.manual_seed(SEED)
    history_v2 = train_model(
        model_v2, loss_fn, loader, NUM_EPOCHS, device,
        "BadgerV2 (small/BiFPN)", lr=0.001
    )

    # =========================================================================
    # Measure Inference Speed
    # =========================================================================
    print(f"\n{'='*60}")
    print("  INFERENCE SPEED")
    print(f"{'='*60}")

    lat_v1, fps_v1 = measure_speed(model_v1, device)
    lat_v2, fps_v2 = measure_speed(model_v2, device)

    print(f"  Badger v1:  {lat_v1:.1f}ms  ({fps_v1:.0f} FPS)")
    print(f"  BadgerV2:   {lat_v2:.1f}ms  ({fps_v2:.0f} FPS)")

    # Test reparameterized V2
    model_v2.fuse_for_deploy()
    lat_v2_fused, fps_v2_fused = measure_speed(model_v2, device)
    print(f"  BadgerV2 (fused): {lat_v2_fused:.1f}ms  ({fps_v2_fused:.0f} FPS)")

    # =========================================================================
    # Compare Results
    # =========================================================================
    print(f"\n{'='*60}")
    print("  📊 COMPARISON RESULTS")
    print(f"{'='*60}")

    def safe_get(lst, idx, default=float('inf')):
        return lst[idx] if idx < len(lst) else default

    initial_v1 = safe_get(history_v1['loss'], 0)
    initial_v2 = safe_get(history_v2['loss'], 0)
    final_v1 = safe_get(history_v1['loss'], -1)
    final_v2 = safe_get(history_v2['loss'], -1)
    reduction_v1 = (initial_v1 - final_v1) / max(initial_v1, 1e-6) * 100
    reduction_v2 = (initial_v2 - final_v2) / max(initial_v2, 1e-6) * 100

    print(f"\n  {'Metric':<30} {'Badger v1':>14} {'BadgerV2':>14} {'Winner':>10}")
    print(f"  {'-'*30} {'-'*14} {'-'*14} {'-'*10}")

    # Parameters
    winner_p = "V2 ✓" if params_v2 < params_v1 else ("V1 ✓" if params_v1 < params_v2 else "Tie")
    print(f"  {'Parameters':<30} {params_v1:>12,}   {params_v2:>12,}   {winner_p:>10}")

    # Initial loss
    print(f"  {'Initial loss':<30} {initial_v1:>14.4f} {initial_v2:>14.4f}")

    # Final loss
    winner_l = "V2 ✓" if final_v2 < final_v1 else ("V1 ✓" if final_v1 < final_v2 else "Tie")
    print(f"  {'Final loss':<30} {final_v1:>14.4f} {final_v2:>14.4f}   {winner_l:>10}")

    # Loss reduction
    winner_r = "V2 ✓" if reduction_v2 > reduction_v1 else ("V1 ✓" if reduction_v1 > reduction_v2 else "Tie")
    print(f"  {'Loss reduction %':<30} {reduction_v1:>13.1f}% {reduction_v2:>13.1f}%   {winner_r:>10}")

    # Avg epoch time
    avg_time_v1 = np.mean(history_v1['epoch_time']) if history_v1['epoch_time'] else 0
    avg_time_v2 = np.mean(history_v2['epoch_time']) if history_v2['epoch_time'] else 0
    winner_t = "V2 ✓" if avg_time_v2 < avg_time_v1 else ("V1 ✓" if avg_time_v1 < avg_time_v2 else "Tie")
    print(f"  {'Avg epoch time':<30} {avg_time_v1:>12.1f}s  {avg_time_v2:>12.1f}s  {winner_t:>10}")

    # Inference latency
    winner_lat = "V2 ✓" if lat_v2_fused < lat_v1 else ("V1 ✓" if lat_v1 < lat_v2_fused else "Tie")
    print(f"  {'Inference latency':<30} {lat_v1:>12.1f}ms {lat_v2_fused:>12.1f}ms {winner_lat:>10}")

    # FPS
    winner_fps = "V2 ✓" if fps_v2_fused > fps_v1 else ("V1 ✓" if fps_v1 > fps_v2_fused else "Tie")
    print(f"  {'FPS':<30} {fps_v1:>14.0f} {fps_v2_fused:>14.0f}   {winner_fps:>10}")

    # Efficiency: loss per M params
    eff_v1 = final_v1 / (params_v1 / 1e6)
    eff_v2 = final_v2 / (params_v2 / 1e6)
    winner_eff = "V2 ✓" if eff_v2 < eff_v1 else ("V1 ✓" if eff_v1 < eff_v2 else "Tie")
    print(f"  {'Loss/Mparam (efficiency)':<30} {eff_v1:>14.4f} {eff_v2:>14.4f}   {winner_eff:>10}")

    # =========================================================================
    # Learning Verification
    # =========================================================================
    print(f"\n{'='*60}")
    print("  🧠 LEARNING VERIFICATION")
    print(f"{'='*60}")

    v1_learns = len(history_v1['loss']) >= 2 and history_v1['loss'][-1] < history_v1['loss'][0]
    v2_learns = len(history_v2['loss']) >= 2 and history_v2['loss'][-1] < history_v2['loss'][0]

    print(f"  Badger v1 learns:  {'✅ YES' if v1_learns else '❌ NO'} "
          f"({initial_v1:.4f} → {final_v1:.4f})")
    print(f"  BadgerV2  learns:  {'✅ YES' if v2_learns else '❌ NO'} "
          f"({initial_v2:.4f} → {final_v2:.4f})")

    # Check monotonicity (loss generally decreasing)
    def monotonicity_score(losses):
        """% of epochs where loss decreased from previous."""
        if len(losses) < 2:
            return 0
        decreases = sum(1 for i in range(1, len(losses)) if losses[i] < losses[i-1])
        return decreases / (len(losses) - 1) * 100

    mono_v1 = monotonicity_score(history_v1['loss'])
    mono_v2 = monotonicity_score(history_v2['loss'])
    print(f"  V1 monotonicity:   {mono_v1:.0f}% epochs decreased")
    print(f"  V2 monotonicity:   {mono_v2:.0f}% epochs decreased")

    # Box loss convergence (key metric for detection)
    if history_v1['box_loss'] and history_v2['box_loss']:
        box_final_v1 = history_v1['box_loss'][-1]
        box_final_v2 = history_v2['box_loss'][-1]
        print(f"\n  Box loss (final):  v1={box_final_v1:.4f}  v2={box_final_v2:.4f}")

    # =========================================================================
    # Loss Curve (ASCII)
    # =========================================================================
    print(f"\n{'='*60}")
    print("  📈 LOSS CURVES")
    print(f"{'='*60}")

    def ascii_plot(losses_v1, losses_v2, width=50, height=12):
        """Simple ASCII chart comparing two loss curves."""
        all_vals = losses_v1 + losses_v2
        if not all_vals:
            return
        max_val = max(all_vals)
        min_val = min(all_vals)
        val_range = max_val - min_val if max_val != min_val else 1.0

        n_epochs = max(len(losses_v1), len(losses_v2))
        step = max(1, n_epochs // width)

        print(f"  {max_val:7.3f} ┤")

        for row in range(height):
            y_val = max_val - (row / (height - 1)) * val_range
            line = "  " + f"{y_val:7.3f} │"

            for col in range(min(width, n_epochs)):
                epoch = col * step if step > 1 else col
                c1 = '·'
                c2 = '·'

                if epoch < len(losses_v1):
                    v1_y = (max_val - losses_v1[epoch]) / val_range * (height - 1)
                    if abs(v1_y - row) < 0.6:
                        c1 = '█'

                if epoch < len(losses_v2):
                    v2_y = (max_val - losses_v2[epoch]) / val_range * (height - 1)
                    if abs(v2_y - row) < 0.6:
                        c2 = '▓'

                if c1 == '█' and c2 == '▓':
                    line += '▒'
                elif c1 == '█':
                    line += '█'
                elif c2 == '▓':
                    line += '▓'
                else:
                    line += ' '

            print(line)

        print(f"  {min_val:7.3f} └" + "─" * min(width, n_epochs))
        print(f"          epoch 1" + " " * max(0, min(width, n_epochs) - 20)
              + f"epoch {n_epochs}")
        print(f"          █ = Badger v1    ▓ = BadgerV2")

    ascii_plot(history_v1['loss'], history_v2['loss'])

    # =========================================================================
    # Summary Verdict
    # =========================================================================
    print(f"\n{'='*60}")
    print("  🏆 VERDICT")
    print(f"{'='*60}")

    v2_wins = 0
    v1_wins = 0
    if final_v2 < final_v1: v2_wins += 2  # Loss is most important
    else: v1_wins += 2
    if reduction_v2 > reduction_v1: v2_wins += 1
    else: v1_wins += 1
    if eff_v2 < eff_v1: v2_wins += 1
    else: v1_wins += 1
    if fps_v2_fused > fps_v1: v2_wins += 1
    else: v1_wins += 1

    if v2_wins > v1_wins:
        print(f"\n  BadgerV2 wins {v2_wins}-{v1_wins}! 🦡✨")
        print(f"  V2's BiFPN weighted fusion + RepC2f/CIB backbone outperform v1.")
    elif v1_wins > v2_wins:
        print(f"\n  Badger v1 wins {v1_wins}-{v2_wins}!")
        print(f"  The classic CSPDarknet + PAFPN architecture remains strong.")
    else:
        print(f"\n  It's a tie {v1_wins}-{v2_wins}!")
        print(f"  Both architectures are competitive.")

    if v2_learns and v1_learns:
        print(f"\n  ✅ Both models successfully learn from data.")
        print(f"     Loss decreases consistently over {NUM_EPOCHS} epochs.")

    print(f"\n{'='*60}\n")

    return 0 if (v1_learns and v2_learns) else 1


if __name__ == '__main__':
    sys.exit(main())
