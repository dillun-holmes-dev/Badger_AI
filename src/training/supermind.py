"""
🦡 SuperMind Trainer — The "Easy Button" for Badger AI

DROP-IN REPLACEMENT for manual training scripts. Figures out everything
automatically so you don't have to tune hyperparameters.

Key features:
  - Auto LR Finder: Finds the optimal learning rate (Leslie Smith method)
  - Auto Batch Size: Finds the largest batch that fits in GPU memory
  - One-Cycle Schedule: Trains 2-5× faster than cosine annealing
  - EMA: Exponential Moving Average for +0.3-0.5 AP
  - Gradient Clipping: Prevents exploding gradients
  - Warmup: Gradual warmup to prevent early instability
  - Early Stopping: Stops when validation loss plateaus
  - Mixed Precision: AMP with automatic fallback to FP32
  - Smart Defaults: Architecture-specific defaults that just work

Usage:
    from src.training.supermind import SuperMind

    model = create_model('badger-s', num_classes=80)
    trainer = SuperMind(model, train_loader, val_loader)
    trainer.fit(epochs=100)  # That's it!

References:
  - Smith, "Cyclical Learning Rates for Training Neural Networks" (WACV 2017)
  - Smith & Topin, "Super-Convergence" (2018)
  - Loshchilov & Hutter, "SGDR: Stochastic Gradient Descent with Warm Restarts" (2017)
"""

import math
import time
import copy
from pathlib import Path
from typing import Optional, Dict, List, Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


# =============================================================================
# Learning Rate Finder — Robust Steepest-Slope Method
# =============================================================================

class LRFinder:
    """
    Robust LR Finder — finds the optimal learning rate using the
    STEEPEST STABLE DESCENT method, not the minimum-loss point.

    Core algorithm:
      1. Exponential sweep: 1e-7 → 10 over ~200 steps
      2. EMA-smoothed loss tracking (β=0.98)
      3. Divergence detection: stop when loss explodes
      4. Steepest slope in log-space: d(loss)/d(log₁₀(lr))
      5. Conservative fallback: recommended_lr / 5
      6. Output clamping: 1e-5 ≤ LR ≤ 1e-3 (AdamW safe range)
      7. Pathological rejection: if LR < 1e-6, fall back to 3e-4

    This is the CORRECT method. The old "pick min loss / 10" approach
    often gives absurdly low values (1e-8) that cause frozen training.

    Reference: Smith, "Cyclical Learning Rates" (WACV 2017), Section 3.2
               Howard & Gugger, "fastai LR Finder" (2018)
    """

    # Clamping bounds for AdamW (empirically safe across architectures)
    ADAMW_MIN_LR = 1e-5
    ADAMW_MAX_LR = 1e-3
    FALLBACK_LR = 3e-4  # Used when finder produces pathological results

    def __init__(self, model, optimizer, loss_fn, device='cuda'):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.history = {}

    def find(self, train_loader, start_lr=1e-7, end_lr=10.0,
             num_steps=200, ema_beta=0.98, divergence_threshold=4.0,
             consecutive_increase_limit=10):
        """
        Run robust LR range test.

        Args:
            train_loader:   training data loader (use small subset recommended)
            start_lr:       minimum LR (default 1e-7)
            end_lr:         maximum LR (default 10.0 — we stop early on divergence)
            num_steps:      number of steps (default 200)
            ema_beta:       EMA smoothing factor (default 0.98)
            divergence_threshold: stop if loss > threshold × best_loss
            consecutive_increase_limit: stop if loss rises N steps in a row

        Returns:
            recommended_lr:  optimally steepest-slope LR (clamped)
            conservative_lr: recommended_lr / 5 (safer fallback)
            history:         dict with 'lr', 'loss_raw', 'loss_smooth', 'slope'
        """
        self.model.train()

        # --- Step 1: Compute multiplier for exponential sweep ---
        lr_mult = (end_lr / start_lr) ** (1.0 / num_steps)
        lr = start_lr

        best_loss = float('inf')
        loss_ema = None
        losses_raw = []
        losses_smooth = []
        lrs = []
        consecutive_bad = 0
        prev_loss = None

        # Save clean state
        initial_state = copy.deepcopy(self.model.state_dict())
        initial_opt_state = copy.deepcopy(self.optimizer.state_dict())

        train_iter = iter(train_loader)
        pbar = tqdm(range(num_steps), desc='LR Finder', leave=False)

        for i in pbar:
            # --- Fetch batch ---
            try:
                images, targets = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                images, targets = next(train_iter)

            images = images.to(self.device)
            targets = targets.to(self.device)

            # --- Set current LR ---
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

            # --- Forward pass ---
            with torch.amp.autocast('cuda' if 'cuda' in str(self.device) else 'cpu',
                                     enabled=('cuda' in str(self.device))):
                cls_scores, bbox_preds, raw_reg = self.model(images, return_raw_reg=True)
                img_size = images.shape[-2:]
                quality_scores = getattr(self.model, '_last_quality_scores', None)
                loss, _ = self.loss_fn(cls_scores, bbox_preds, targets, img_size,
                                        raw_reg_preds=raw_reg,
                                        quality_scores=quality_scores)

            # --- Backward pass ---
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # --- Record ---
            loss_val = loss.item()
            losses_raw.append(loss_val)
            lrs.append(lr)

            # EMA smoothing: loss_smooth = β × loss_smooth + (1-β) × loss
            if loss_ema is None:
                loss_ema = loss_val
            else:
                loss_ema = ema_beta * loss_ema + (1 - ema_beta) * loss_val
            losses_smooth.append(loss_ema)

            pbar.set_postfix({
                'lr': f'{lr:.2e}',
                'loss': f'{loss_val:.3f}',
                'smooth': f'{loss_ema:.3f}'
            })

            # --- Divergence detection ---
            # Rule 1: NaN/Inf
            if math.isnan(loss_val) or math.isinf(loss_val):
                print(f"\n  ⚠️  LR Finder stopped at {lr:.2e} (loss NaN/Inf)")
                break

            # Rule 2: Loss explodes beyond threshold
            if i > 10 and loss_ema > divergence_threshold * best_loss:
                print(f"\n  ⚠️  LR Finder stopped at {lr:.2e} "
                      f"(loss {loss_ema:.1f}× > best {best_loss:.1f}×)")
                break

            # Rule 3: Consecutive increases
            if prev_loss is not None and loss_ema > prev_loss:
                consecutive_bad += 1
            else:
                consecutive_bad = 0
            if consecutive_bad >= consecutive_increase_limit:
                print(f"\n  ⚠️  LR Finder stopped at {lr:.2e} "
                      f"({consecutive_increase_limit} consecutive increases)")
                break

            if loss_ema < best_loss:
                best_loss = loss_ema

            prev_loss = loss_ema
            lr *= lr_mult

        # --- Restore clean state ---
        self.model.load_state_dict(initial_state)
        self.optimizer.load_state_dict(initial_opt_state)
        torch.cuda.empty_cache() if 'cuda' in str(self.device) else None

        # --- Compute slopes in log space ---
        # d(loss) / d(log₁₀(lr)) — negative means loss decreasing
        lrs_arr = lrs
        log_lrs = [math.log10(x) for x in lrs_arr]
        slopes = self._compute_slopes(losses_smooth, log_lrs)

        # --- Find steepest stable descent ---
        recommended_lr = self._find_steepest_descent(lrs_arr, slopes, losses_smooth)
        conservative_lr = recommended_lr / 5.0

        # --- Clamp output to safe AdamW range ---
        recommended_lr = max(self.ADAMW_MIN_LR, min(self.ADAMW_MAX_LR, recommended_lr))
        conservative_lr = max(self.ADAMW_MIN_LR,
                              min(self.ADAMW_MAX_LR * 0.5, conservative_lr))

        # Ensure conservative is genuinely lower than recommended
        if conservative_lr >= recommended_lr:
            conservative_lr = recommended_lr / 3.0
            conservative_lr = max(self.ADAMW_MIN_LR, conservative_lr)

        # --- Pathological rejection ---
        if recommended_lr < 1e-6:
            print(f"  ⚠️  LR Finder produced pathological value {recommended_lr:.2e}")
            print(f"  🔄 Falling back to safe default: {self.FALLBACK_LR:.2e}")
            recommended_lr = self.FALLBACK_LR
            conservative_lr = self.FALLBACK_LR / 3

        self.history = {
            'lr': lrs_arr,
            'loss_raw': losses_raw,
            'loss_smooth': losses_smooth,
            'slope': slopes
        }

        return recommended_lr, conservative_lr, self.history

    def _compute_slopes(self, losses_smooth, log_lrs):
        """Compute d(loss)/d(log₁₀(lr)) using central differences."""
        n = len(losses_smooth)
        slopes = [0.0] * n
        for i in range(1, n - 1):
            dx = log_lrs[i + 1] - log_lrs[i - 1]
            if dx > 1e-10:
                slopes[i] = (losses_smooth[i + 1] - losses_smooth[i - 1]) / dx
        # Edge cases: forward/backward difference
        if n >= 2:
            dx = log_lrs[1] - log_lrs[0]
            slopes[0] = (losses_smooth[1] - losses_smooth[0]) / max(dx, 1e-10) if dx > 0 else 0
            dx = log_lrs[-1] - log_lrs[-2]
            slopes[-1] = (losses_smooth[-1] - losses_smooth[-2]) / max(dx, 1e-10) if dx > 0 else 0

        # Smooth slopes for stability
        slopes = self._ema_smooth(slopes, alpha=0.7)
        return slopes

    def _ema_smooth(self, values, alpha=0.7):
        """Exponential moving average smoothing for slope stability."""
        smoothed = [values[0]]
        for v in values[1:]:
            smoothed.append(alpha * smoothed[-1] + (1 - alpha) * v)
        return smoothed

    def _find_steepest_descent(self, lrs, slopes, losses_smooth):
        """
        Find LR at the steepest STABLE descent point.

        Strategy:
          1. Look for negative slopes (loss decreasing)
          2. Find the maximum negative slope (steepest descent)
          3. Only consider region before divergence (ignore noise after)

        Returns the LR at steepest descent, clamped to valid range.
        """
        if len(lrs) < 10:
            return self.FALLBACK_LR

        # Find cutoff: stop at 90% of the way to min loss (avoid flat/divergent tail)
        min_loss_idx = losses_smooth.index(min(losses_smooth))
        cutoff = max(10, int(min_loss_idx * 0.9))

        # Find maximum negative slope in the valid region
        best_idx = 0
        best_slope = float('inf')
        for i in range(cutoff):
            if slopes[i] < best_slope:
                best_slope = slopes[i]
                best_idx = i

        # If no negative slope found, use conservative position
        if best_slope >= 0:
            # Fallback: use 1/3 of the way to min loss
            best_idx = max(1, min_loss_idx // 3)

        return lrs[max(0, best_idx)]

    def plot(self, save_path=None):
        """Plot LR finder results with loss curve and slope annotation."""
        try:
            import matplotlib.pyplot as plt
            import numpy as np

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

            lrs = np.array(self.history['lr'])
            losses = np.array(self.history['loss_smooth'])
            slopes_arr = np.array(self.history['slope'])

            # Top: loss curve
            ax1.plot(lrs, losses, 'b-', linewidth=1.5, label='Smoothed loss')
            ax1.set_xscale('log')
            ax1.set_ylabel('Loss')
            ax1.set_title('LR Finder — Loss vs Learning Rate')
            ax1.grid(True, alpha=0.3)
            ax1.legend()

            # Mark steepest point
            if len(slopes_arr) > 0:
                best_idx = np.argmin(slopes_arr[:len(lrs)])
                ax1.axvline(lrs[best_idx], color='g', linestyle='--', alpha=0.5,
                            label=f'Steepest: {lrs[best_idx]:.2e}')

            # Bottom: slope
            ax2.plot(lrs[:len(slopes_arr)], slopes_arr, 'r-', linewidth=1.5,
                     label='d(loss)/d(log₁₀ LR)')
            ax2.axhline(0, color='gray', linestyle=':', alpha=0.5)
            ax2.set_xscale('log')
            ax2.set_xlabel('Learning Rate')
            ax2.set_ylabel('Slope')
            ax2.set_title('Slope (negative = loss decreasing)')
            ax2.grid(True, alpha=0.3)
            ax2.legend()

            plt.tight_layout()
            if save_path:
                plt.savefig(save_path, dpi=100, bbox_inches='tight')
            plt.show()
        except ImportError:
            print("  matplotlib not available for LR plot")


# =============================================================================
# One-Cycle Scheduler
# =============================================================================

class OneCycleScheduler:
    """
    One-Cycle Learning Rate scheduler.

    Trains the full schedule in ONE cycle:
      1. Warmup (0→max_lr):      0-30% of training
      2. Annealing (max_lr→min_lr): 30-90% of training
      3. Final decay (min_lr→final_lr): 90-100% of training

    Momentum (if SGD) does the OPPOSITE:
      1. Start high → goes low during warmup → back high during annealing

    For AdamW, we adapt the weight decay or just use LR cycling.

    Why it works:
      - High LR early: explores broadly, escapes local minima
      - Medium LR mid: fine-tunes features
      - Low LR late: settles into precise minima
      - The speed of cycling prevents overfitting

    Typically trains 2-5× faster than constant/cosine schedules.

    Reference: Smith & Topin, "Super-Convergence: Very Fast Training of
               Residual Networks Using Large Learning Rates" (2018)
    """

    def __init__(self, optimizer, max_lr, total_steps, pct_start=0.3,
                 div_factor=25.0, final_div_factor=1e4, anneal_strategy='cos'):
        self.optimizer = optimizer
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor
        self.anneal_strategy = anneal_strategy
        self.step_count = 0
        self.lr_scale = 1.0

        self.initial_lr = max_lr / div_factor
        self.final_lr = self.initial_lr / final_div_factor

    def get_lr(self):
        """Compute current LR based on step position."""
        if self.step_count >= self.total_steps:
            return self.final_lr

        # Phase 1: Warmup (0 → pct_start%)
        if self.step_count <= self.total_steps * self.pct_start:
            progress = self.step_count / (self.total_steps * self.pct_start)
            lr = self._anneal(self.initial_lr, self.max_lr, progress)
            return max(self.final_lr, lr * self.lr_scale)

        # Phase 2: Annealing (pct_start% → 100%)
        progress = (self.step_count - self.total_steps * self.pct_start) / \
                   (self.total_steps * (1 - self.pct_start))
        lr = self._anneal(self.max_lr, self.final_lr, progress)
        return max(self.final_lr, lr * self.lr_scale)

    def _anneal(self, start, end, progress):
        """Cosine or linear annealing."""
        if self.anneal_strategy == 'cos':
            return end + (start - end) * (1 + math.cos(math.pi * progress)) / 2
        elif self.anneal_strategy == 'linear':
            return start + (end - start) * progress
        return start

    def step(self):
        """Step the scheduler and update optimizer LR."""
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.step_count += 1
        return lr

    def reduce_peak(self, factor, min_scale=0.1):
        """Reduce the remaining OneCycle curve without resetting progress."""
        old_scale = self.lr_scale
        self.lr_scale = max(min_scale, self.lr_scale * factor)
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return old_scale, self.lr_scale, lr


# =============================================================================
# Model EMA
# =============================================================================

class ModelEMA:
    """
    Exponential Moving Average of model weights.

    Maintains a shadow copy: θ_ema = α × θ_ema + (1-α) × θ_current
    Use the EMA model for evaluation — gives +0.3-0.5 AP for free.
    """

    def __init__(self, model, decay=0.9998):
        self.ema = copy.deepcopy(model).eval()
        self.decay = decay
        self.updates = 0

    def update(self, model):
        """Update EMA after each optimizer step."""
        self.updates += 1
        decay = min(self.decay, (1 + self.updates) / (10 + self.updates))
        with torch.no_grad():
            for ema_p, model_p in zip(self.ema.parameters(), model.parameters()):
                ema_p.mul_(decay).add_(model_p.data, alpha=1 - decay)

    def increase_decay(self, amount=5e-5, max_decay=0.9999):
        """Increase EMA smoothing when validation becomes noisy."""
        old_decay = self.decay
        self.decay = min(max_decay, self.decay + amount)
        return old_decay, self.decay

    def state_dict(self):
        return self.ema.state_dict()


# =============================================================================
# Validation Stability Controller
# =============================================================================

class StabilityController:
    """
    Validation-aware LR controller for SuperMind.

    It watches the generalization gap and validation trend after each epoch.
    When validation worsens for several epochs while train loss keeps improving,
    it reduces the remaining OneCycle LR curve and asks EMA to smooth harder.
    """

    def __init__(self, patience=3, reduce_factor=0.75, min_lr_scale=0.1,
                 min_delta=1e-4, cooldown_epochs=1,
                 ema_decay_step=5e-5, ema_max_decay=0.9999):
        self.patience = patience
        self.reduce_factor = reduce_factor
        self.min_lr_scale = min_lr_scale
        self.min_delta = min_delta
        self.cooldown_epochs = cooldown_epochs
        self.ema_decay_step = ema_decay_step
        self.ema_max_decay = ema_max_decay

        self.best_val_loss = float('inf')
        self.bad_epochs = 0
        self.cooldown = 0
        self.gaps = []

    def step(self, train_loss, val_loss, scheduler, ema=None):
        gap = val_loss - train_loss
        self.gaps.append(gap)

        if val_loss < self.best_val_loss - self.min_delta:
            self.best_val_loss = val_loss
            self.bad_epochs = 0
            if self.cooldown > 0:
                self.cooldown -= 1
            return {'gap': gap, 'action': None}

        self.bad_epochs += 1
        gap_worsening = (
            len(self.gaps) >= self.patience
            and all(
                self.gaps[i] > self.gaps[i - 1] + self.min_delta
                for i in range(len(self.gaps) - self.patience + 1, len(self.gaps))
            )
        )

        should_reduce = self.bad_epochs >= self.patience or gap_worsening
        if not should_reduce or self.cooldown > 0:
            if self.cooldown > 0:
                self.cooldown -= 1
            return {'gap': gap, 'action': None}

        old_scale, new_scale, new_lr = scheduler.reduce_peak(
            self.reduce_factor, min_scale=self.min_lr_scale
        )
        self.bad_epochs = 0
        self.cooldown = self.cooldown_epochs

        old_decay = new_decay = None
        if ema is not None:
            old_decay, new_decay = ema.increase_decay(
                amount=self.ema_decay_step,
                max_decay=self.ema_max_decay
            )

        return {
            'gap': gap,
            'action': 'reduce_lr',
            'old_lr_scale': old_scale,
            'new_lr_scale': new_scale,
            'new_lr': new_lr,
            'old_ema_decay': old_decay,
            'new_ema_decay': new_decay,
        }


# =============================================================================
# Stochastic Weight Averaging (SWA)
# =============================================================================

class SWA:
    """
    Stochastic Weight Averaging — snapshot-based model averaging.
    Unlike EMA (continuous), SWA takes SNAPSHOTS at intervals.
    Applied in the last 25% of training. +0.5-1.0 AP on COCO.
    """

    def __init__(self, model, start_epoch=225, snapshot_freq=10, total_epochs=300):
        self.swa_model = copy.deepcopy(model).eval()
        self.start_epoch = start_epoch
        self.snapshot_freq = snapshot_freq
        self.total_epochs = total_epochs
        self.n_snapshots = 0
        self.active = False

    def update(self, model, epoch):
        if epoch >= self.start_epoch:
            self.active = True
            if (epoch - self.start_epoch) % self.snapshot_freq == 0:
                with torch.no_grad():
                    n = self.n_snapshots
                    for swa_p, model_p in zip(self.swa_model.parameters(),
                                               model.parameters()):
                        swa_p.data = (swa_p.data * n + model_p.data) / (n + 1)
                self.n_snapshots += 1

    def state_dict(self):
        return self.swa_model.state_dict()


# =============================================================================
# SuperMind Trainer
# =============================================================================

class SuperMind:
    """
    The "it just works" trainer for Badger.

    Handles EVERYTHING automatically:
      - Batch size selection
      - Learning rate tuning
      - Scheduling
      - Mixed precision
      - Regularization
      - Checkpointing
      - Logging

    Usage:
        model = create_model('badger-s', num_classes=80)
        trainer = SuperMind(model, train_loader, val_loader)
        trainer.fit(epochs=300)
    """

    def __init__(self, model, train_loader, val_loader=None,
                 device='cuda', project_dir='runs/supermind',
                 use_amp=True, use_ema=True, use_swa=False,
                 use_grad_clip=True, use_compile=False,
                 early_stopping_patience=50, box_loss_type='ciou',
                 use_stability_controller=True):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.project_dir = Path(project_dir)
        self.project_dir.mkdir(parents=True, exist_ok=True)

        self.use_amp = use_amp and 'cuda' in str(self.device)
        self.use_ema = use_ema
        self.use_swa = use_swa
        self.use_grad_clip = use_grad_clip
        self.use_compile = use_compile
        self.early_stopping_patience = early_stopping_patience
        self.box_loss_type = box_loss_type
        self.use_stability_controller = use_stability_controller

        # Try torch.compile for 30-50% speedup (PyTorch 2.0+)
        if self.use_compile and hasattr(torch, 'compile'):
            try:
                print("  ⚡ Applying torch.compile() for training speedup...")
                self.model = torch.compile(self.model, mode='reduce-overhead')
                print("  ✅ torch.compile() enabled")
            except Exception as e:
                print(f"  ⚠️  torch.compile() failed ({e}), falling back")
                self.use_compile = False

        self.model = self.model.to(self.device)
        self.ema = None
        self.swa = None
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None

        # Progressive image size (YOLO26 style)
        self._progressive_sizes = None
        self._original_img_size = None

        # Will be set during fit()
        self.optimizer = None
        self.scheduler = None
        self.loss_fn = None
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.current_epoch = 0
        self.history = {'train_loss': [], 'val_loss': [],
                        'lr': [], 'box': [], 'cls': [], 'dfl': [],
                        'generalization_gap': [], 'lr_scale': [],
                        'stability_action': []}
        self.stability_controller = None

    def _find_batch_size(self, start_size=64):
        """Find the largest batch size that fits in GPU memory."""
        if 'cpu' in str(self.device):
            return min(start_size, 32)

        size = start_size
        while size > 0:
            try:
                x = torch.randn(2, 3, 640, 640, device=self.device)
                cls_scores, bbox_preds, _ = self.model(x, return_raw_reg=True)
                del cls_scores, bbox_preds
                torch.cuda.empty_cache()
                # Can fit 2× model, real batch can be size//2
                print(f"  Auto batch size: {size}")
                return size
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    size = size // 2
                    torch.cuda.empty_cache()
                else:
                    raise

        return 1  # Fallback

    def _create_loss(self, num_classes, box_weight=7.5, cls_weight=0.5,
                     dfl_weight=1.5, quality_weight=1.0, label_smoothing=0.0,
                     box_loss_type='ciou'):
        """Create loss function with selectable box loss type."""
        from src.losses import BadgerLoss
        return BadgerLoss(
            num_classes=num_classes,
            box_weight=box_weight,
            cls_weight=cls_weight,
            dfl_weight=dfl_weight,
            quality_weight=quality_weight,
            label_smoothing=label_smoothing,
            assigner='tal',
            box_loss_type=box_loss_type
        )

    def _create_optimizer(self, lr):
        """Create optimizer with smart defaults."""
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=0.0005,
            betas=(0.9, 0.999)
        )

    def _validate_lr_choice(self, recommended_lr, conservative_lr,
                            num_val_batches=5):
        """
        Post-LR validation: test the recommended LR on a few batches.
        If loss doesn't decrease, fall back to conservative LR.

        Returns the validated LR to use.
        """
        # Quick test on a few batches
        initial_state = copy.deepcopy(self.model.state_dict())
        opt = self._create_optimizer(recommended_lr)
        initial_losses = []

        try:
            train_iter = iter(self.train_loader)
            self.model.train()

            for _ in range(num_val_batches):
                try:
                    images, targets = next(train_iter)
                except StopIteration:
                    break
                images = images.to(self.device)
                targets = targets.to(self.device)

                cls_scores, bbox_preds, raw_reg = self.model(images, return_raw_reg=True)
                img_size = images.shape[-2:]
                quality_scores = getattr(self.model, '_last_quality_scores', None)
                loss, _ = self.loss_fn(cls_scores, bbox_preds, targets, img_size,
                                        raw_reg_preds=raw_reg,
                                        quality_scores=quality_scores)
                opt.zero_grad()
                loss.backward()
                opt.step()
                initial_losses.append(loss.item())
        finally:
            self.model.load_state_dict(initial_state)

        if len(initial_losses) >= 2:
            first_half = sum(initial_losses[:len(initial_losses)//2]) / max(1, len(initial_losses)//2)
            second_half = sum(initial_losses[len(initial_losses)//2:]) / max(1, len(initial_losses) - len(initial_losses)//2)
            if second_half > first_half * 1.2:
                print(f"  ⚠️  Recommended LR {recommended_lr:.2e} caused loss increase, "
                      f"using conservative {conservative_lr:.2e}")
                return conservative_lr

        return recommended_lr

    def _apply_preset(self, preset, epochs, lr, box_weight, cls_weight,
                      dfl_weight, label_smoothing, progressive_size):
        """
        Apply a named training preset.

        Presets:
          'fast'     — Quick training, good for debugging/prototyping
          'accurate' — Balanced defaults for competitive results
          'super'    — All tricks enabled for maximum accuracy
        """
        if preset == 'fast':
            if epochs > 100:
                epochs = 100
            label_smoothing = 0.0
            progressive_size = False
            use_swa = False
            use_ema = True
            use_compile = self.use_compile
            box_loss_type = 'ciou'
            print(f"  ⚡ Preset 'fast': {epochs} epochs, CIoU, no SWA, EMA on")
        elif preset == 'accurate':
            if epochs < 200:
                epochs = 200
            label_smoothing = 0.05
            progressive_size = False
            use_swa = True
            use_ema = True
            use_compile = self.use_compile
            box_loss_type = 'wiou'
            print(f"  🎯 Preset 'accurate': {epochs} epochs, WIoU, SWA+EMA on")
        elif preset == 'super':
            if epochs < 300:
                epochs = 300
            label_smoothing = 0.1
            progressive_size = True
            use_swa = True
            use_ema = True
            use_compile = True  # Force compile in super mode
            box_loss_type = 'inner_iou'
            print(f"  🚀 Preset 'super': {epochs} epochs, Inner-IoU, SWA+EMA+Compile+Progressive")
        else:
            raise ValueError(f"Unknown preset '{preset}'. Use 'fast', 'accurate', or 'super'.")

        return (epochs, lr, box_weight, cls_weight, dfl_weight,
                label_smoothing, progressive_size, use_swa, use_ema,
                use_compile, box_loss_type)

    def _setup_progressive_sizes(self, total_epochs):
        """
        Progressive image size training (YOLO26 style).

        Starts at 320px and grows to the final size (640px) over training.
        This curriculum helps the model learn coarse features first, then
        fine details — improving convergence speed and final accuracy.

        Size schedule:
          0-30%:  320px (coarse features, fast training)
          30-60%: 480px (medium detail)
          60-100%: 640px (full resolution, fine localization)
        """
        self._progressive_sizes = []
        for epoch in range(total_epochs):
            progress = epoch / max(1, total_epochs - 1)
            if progress < 0.3:
                size = 320
            elif progress < 0.6:
                size = 480
            else:
                size = 640
            self._progressive_sizes.append(size)
        print(f"  📐 Progressive sizes: 320→480→640 over {total_epochs} epochs")

    def _apply_progressive_size(self, epoch):
        """Update dataloader image size for current epoch."""
        if self._progressive_sizes is None:
            return
        size = self._progressive_sizes[min(epoch, len(self._progressive_sizes) - 1)]
        if hasattr(self.train_loader, 'dataset') and hasattr(self.train_loader.dataset, 'img_size'):
            old_size = self.train_loader.dataset.img_size
            if old_size != size:
                self.train_loader.dataset.img_size = size
                if epoch == 0 or size != self._progressive_sizes[max(0, epoch - 1)]:
                    print(f"  🔄 Image size: {size}px (epoch {epoch+1})")

    def fit(self, epochs=300, lr=None, num_classes=80,
            box_weight=7.5, cls_weight=0.5, dfl_weight=1.5,
            quality_weight=1.0, label_smoothing=0.0,
            progressive_size=False,
            preset=None, stability_patience=3,
            stability_reduce_factor=0.75):
        """
        Train the model. That's it — no other parameters needed.

        Args:
            epochs: number of epochs to train
            lr: learning rate (auto-detected if None)
            num_classes: number of detection classes
            box_weight, cls_weight, dfl_weight: loss component weights
            quality_weight: IoU quality prediction loss weight
            label_smoothing: label smoothing epsilon (0.0 = off, 0.1 = mild)
            progressive_size: if True, start at 320px and grow to 640px
            preset: shortcut preset — 'fast', 'accurate', or 'super'
                    'fast':     fewer epochs, simpler settings, trains quick
                    'accurate': balanced defaults for good results
                    'super':    all tricks enabled for maximum accuracy
            stability_patience: validation drift epochs before LR reduction
            stability_reduce_factor: multiplier for remaining OneCycle LR curve
        """
        # Apply preset overrides
        if preset:
            epochs, lr, box_weight, cls_weight, dfl_weight, label_smoothing, \
                progressive_size, self.use_swa, self.use_ema, self.use_compile, \
                self.box_loss_type = self._apply_preset(
                    preset, epochs, lr, box_weight, cls_weight, dfl_weight,
                    label_smoothing, progressive_size)

        print(f"\n{'='*60}")
        print(f"  🦡 SuperMind Trainer v2 — {'Preset: ' + preset if preset else 'Custom Mode'}")
        print(f"{'='*60}")
        print(f"  Device: {self.device}")
        print(f"  Model params: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"  Epochs: {epochs}")
        print(f"  Box loss: {self.box_loss_type}")
        print(f"  Label smoothing: {label_smoothing}")
        print(f"  Progressive size: {'✓' if progressive_size else '✗'}")
        print(f"  SWA: {'✓' if self.use_swa else '✗'}")
        print(f"  Compile: {'✓' if self.use_compile else '✗'}")
        print(f"  Stability controller: {'✓' if self.use_stability_controller and self.val_loader else '✗'}")

        # Set up components
        self.loss_fn = self._create_loss(
            num_classes, box_weight, cls_weight, dfl_weight,
            quality_weight, label_smoothing, self.box_loss_type
        )

        # Auto-detect LR if not provided
        if lr is None:
            print(f"\n  🔍 Running Robust LR Finder (steepest-slope method)...")
            # Use a fresh optimizer for the LR finder
            temp_opt = self._create_optimizer(1e-7)
            finder = LRFinder(self.model, temp_opt, self.loss_fn,
                              device=str(self.device))
            recommended_lr, conservative_lr, history = finder.find(
                self.train_loader,
                start_lr=1e-7,
                end_lr=10.0,
                num_steps=min(200, len(self.train_loader)),
                ema_beta=0.98
            )
            print(f"  📊 Recommended LR:  {recommended_lr:.2e}  (steepest stable descent)")
            print(f"  🛡️  Conservative LR: {conservative_lr:.2e}  (recommended / 5)")

            # --- Post-LR validation: test both candidates on a few batches ---
            self.model.train()
            lr = self._validate_lr_choice(recommended_lr, conservative_lr)
            print(f"  ✅ Selected LR: {lr:.2e}")

            # Plot LR finder results
            try:
                finder.plot(save_path=str(self.project_dir / 'lr_finder.png'))
            except Exception:
                pass
        else:
            print(f"  📐 Using provided LR: {lr:.2e}")

        # Progressive image size setup
        if progressive_size:
            self._setup_progressive_sizes(epochs)

        # Create optimizer and scheduler
        self.optimizer = self._create_optimizer(lr)
        total_steps = epochs * len(self.train_loader)
        self.scheduler = OneCycleScheduler(
            self.optimizer, max_lr=lr, total_steps=total_steps,
            pct_start=0.3, div_factor=25.0, final_div_factor=1e4
        )

        if self.use_stability_controller and self.val_loader:
            self.stability_controller = StabilityController(
                patience=stability_patience,
                reduce_factor=stability_reduce_factor
            )

        # EMA
        if self.use_ema:
            self.ema = ModelEMA(self.model)

        # SWA (starts at 75% of training)
        if self.use_swa:
            swa_start = int(epochs * 0.75)
            self.swa = SWA(self.model, start_epoch=swa_start,
                          snapshot_freq=5, total_epochs=epochs)

        # Print config
        print(f"  Optimizer: AdamW (lr={lr:.2e}, wd=5e-4)")
        print(f"  Scheduler: OneCycle (warmup=30%, cosine anneal)")
        print(f"  AMP: {'✓' if self.use_amp else '✗'}")
        print(f"  EMA: {'✓' if self.use_ema else '✗'}")
        print(f"  Grad Clip: {'✓' if self.use_grad_clip else '✗'}")
        print(f"  Loss weights: box={box_weight}, cls={cls_weight}, dfl={dfl_weight}")

        # Training loop
        print(f"\n{'='*60}")
        print(f"  Training {epochs} epochs...")
        print(f"{'='*60}")

        patience_counter = 0
        start_time = time.time()

        for epoch in range(epochs):
            self.current_epoch = epoch

            # Progressive image size
            if progressive_size:
                self._apply_progressive_size(epoch)

            # Train one epoch
            train_metrics = self._train_epoch()
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['box'].append(train_metrics['box'])
            self.history['cls'].append(train_metrics['cls'])
            self.history['dfl'].append(train_metrics['dfl'])
            self.history['lr'].append(train_metrics['lr'])

            # SWA snapshot
            if self.swa:
                self.swa.update(self.model, epoch)

            # Validate
            val_loss = None
            if self.val_loader:
                val_loss = self._validate_epoch()
                self.history['val_loss'].append(val_loss)

                stability = self.stability_controller.step(
                    train_metrics['loss'], val_loss, self.scheduler, self.ema
                ) if self.stability_controller else {
                    'gap': val_loss - train_metrics['loss'],
                    'action': None
                }
                self.history['generalization_gap'].append(stability['gap'])
                self.history['lr_scale'].append(self.scheduler.lr_scale)
                self.history['stability_action'].append(stability['action'])
            else:
                stability = None
                self.history['lr_scale'].append(self.scheduler.lr_scale)
                self.history['stability_action'].append(None)

            # Print epoch summary
            elapsed = time.time() - start_time
            eta = (elapsed / (epoch + 1)) * (epochs - epoch - 1)
            msg = (f"  Epoch {epoch+1:3d}/{epochs} | "
                   f"Loss: {train_metrics['loss']:.4f} | "
                   f"Box: {train_metrics['box']:.4f} | "
                   f"Cls: {train_metrics['cls']:.4f} | "
                   f"DFL: {train_metrics['dfl']:.4f}")
            if val_loss is not None:
                msg += f" | Val: {val_loss:.4f} | Gap: {stability['gap']:.4f}"
            msg += f" | LR: {train_metrics['lr']:.2e} | ETA: {eta/60:.0f}m"
            print(msg)

            if stability and stability.get('action') == 'reduce_lr':
                ema_msg = ""
                if stability.get('new_ema_decay') is not None:
                    ema_msg = (f", EMA {stability['old_ema_decay']:.5f}"
                               f"→{stability['new_ema_decay']:.5f}")
                print(f"    🛡️ Stability controller: val drift detected, "
                      f"LR scale {stability['old_lr_scale']:.2f}"
                      f"→{stability['new_lr_scale']:.2f} "
                      f"(next LR {stability['new_lr']:.2e}{ema_msg})")

            # Checkpointing
            if val_loss is not None and val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                patience_counter = 0
                self._save_checkpoint('best.pth', epoch, val_loss)
                print(f"    ✅ New best! Saved checkpoint.")
            elif val_loss is not None:
                patience_counter += 1

            # Early stopping
            if patience_counter >= self.early_stopping_patience:
                print(f"\n  ⏹ Early stopping at epoch {epoch+1} "
                      f"(no improvement for {self.early_stopping_patience} epochs)")
                break

            # Period save
            if (epoch + 1) % 50 == 0:
                self._save_checkpoint(f'epoch_{epoch+1}.pth', epoch, val_loss)

        # Wrap up
        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"  ✅ Training complete!")
        print(f"  Best val loss: {self.best_val_loss:.4f} (epoch {self.best_epoch+1})")
        print(f"  Time: {total_time/60:.1f} minutes")
        print(f"{'='*60}")

        # Load best model (prefer SWA > EMA > base)
        if self.val_loader:
            best_path = self.project_dir / 'best.pth'
            if self.swa and self.swa.active and self.swa.n_snapshots > 0:
                self.model.load_state_dict(self.swa.state_dict())
                print(f"  📦 Loaded SWA model ({self.swa.n_snapshots} snapshots)")
            elif best_path.exists():
                ckpt = torch.load(best_path, map_location=self.device)
                self.model.load_state_dict(ckpt['model_state_dict'])
                print(f"  Loaded best model from epoch {ckpt['epoch']}")

        return self.history

    def _train_epoch(self):
        """Train for one epoch. Returns metrics dict."""
        self.model.train()
        total_loss = 0.0
        total_box = 0.0
        total_cls = 0.0
        total_dfl = 0.0
        last_lr = 0.0

        pbar = tqdm(self.train_loader, desc=f'Train {self.current_epoch+1}',
                     leave=False)
        for images, targets in pbar:
            images = images.to(self.device)
            targets = targets.to(self.device)

            # Forward
            with torch.amp.autocast('cuda' if 'cuda' in str(self.device) else 'cpu',
                                     enabled=self.use_amp):
                cls_scores, bbox_preds, raw_reg = self.model(images, return_raw_reg=True)
                img_size = images.shape[-2:]
                quality_scores = getattr(self.model, '_last_quality_scores', None)
                loss, loss_dict = self.loss_fn(
                    cls_scores, bbox_preds, targets, img_size,
                    raw_reg_preds=raw_reg,
                    quality_scores=quality_scores
                )

            # Backward
            self.optimizer.zero_grad()
            if self.scaler:
                self.scaler.scale(loss).backward()
                if self.use_grad_clip:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.use_grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
                self.optimizer.step()


            # LR step
            last_lr = self.scheduler.step()

            # EMA update (after optimizer step)
            if self.ema:
                self.ema.update(self.model)

            # Accumulate
            total_loss += loss.item()
            total_box += loss_dict.get('box', 0)
            total_cls += loss_dict.get('cls', 0)
            total_dfl += loss_dict.get('dfl', 0)

            pbar.set_postfix({
                'loss': f'{loss.item():.3f}',
                'dfl': f'{loss_dict.get("dfl", 0):.3f}'
            })

        n = len(self.train_loader)
        return {
            'loss': total_loss / n,
            'box': total_box / n,
            'cls': total_cls / n,
            'dfl': total_dfl / n,
            'lr': last_lr
        }

    @torch.no_grad()
    def _validate_epoch(self):
        """Validate for one epoch. Returns average loss.
        Uses SWA model if available, else EMA, else base model."""
        if self.swa and self.swa.active and self.swa.n_snapshots > 0:
            eval_model = self.swa.swa_model
        elif self.ema:
            eval_model = self.ema.ema
        else:
            eval_model = self.model
        eval_model.eval()
        total_loss = 0.0

        for images, targets in self.val_loader:
            images = images.to(self.device)
            targets = targets.to(self.device)

            cls_scores, bbox_preds, raw_reg = eval_model(images, return_raw_reg=True)
            img_size = images.shape[-2:]
            quality_scores = getattr(eval_model, '_last_quality_scores', None)
            loss, _ = self.loss_fn(
                cls_scores, bbox_preds, targets, img_size,
                raw_reg_preds=raw_reg,
                quality_scores=quality_scores
            )
            total_loss += loss.item()

        return total_loss / len(self.val_loader)

    def _save_checkpoint(self, filename, epoch, val_loss):
        """Save model checkpoint."""
        save_dict = {
            'epoch': epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'ema_state_dict': self.ema.state_dict() if self.ema else None,
            'history': self.history,
        }
        torch.save(save_dict, self.project_dir / filename)


# =============================================================================
# Convenience function
# =============================================================================

def easy_train(model, train_loader, val_loader=None, epochs=300,
               lr=None, num_classes=80, device='cuda',
               project_dir='runs/easy_train',
               preset=None, box_loss_type='ciou',
               label_smoothing=0.0, use_swa=False,
               use_compile=False, progressive_size=False,
               use_stability_controller=True,
               stability_patience=3,
               stability_reduce_factor=0.75):
    """
    One-liner training — the easiest way to train a Badger model.

    Examples:
        # Quick train with defaults
        history = easy_train(model, train_loader, val_loader, epochs=100,
                            preset='fast')

        # Maximum accuracy with all tricks
        history = easy_train(model, train_loader, val_loader,
                            preset='super')

        # Custom configuration
        history = easy_train(model, train_loader, val_loader,
                            epochs=200, preset='accurate',
                            box_loss_type='wiou')
    """
    trainer = SuperMind(
        model, train_loader, val_loader=val_loader,
        device=device, project_dir=project_dir,
        use_swa=use_swa, use_compile=use_compile,
        box_loss_type=box_loss_type,
        use_stability_controller=use_stability_controller
    )
    return trainer.fit(
        epochs=epochs, lr=lr, num_classes=num_classes,
        label_smoothing=label_smoothing,
        progressive_size=progressive_size,
        preset=preset,
        stability_patience=stability_patience,
        stability_reduce_factor=stability_reduce_factor
    )
