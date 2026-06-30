"""
Building blocks used throughout the Badger architecture.

Every Badger model is built from these fundamental pieces:
  - Conv:      Standard conv + batch norm + activation
  - Bottleneck: Residual block with two convs
  - C2f:       Badger's cross-stage partial fusion (replaces C3)
  - SPPF:      Spatial pyramid pooling — fast variant
  - Detect:    The detection head that produces final predictions
"""

import torch
import torch.nn as nn
import math


def autopad(k, p=None):
    """Auto-calculate padding to keep spatial dimensions same for odd kernels."""
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


# =============================================================================
# 1. Conv — the basic building block
# =============================================================================
class Conv(nn.Module):
    """
    Standard convolution block: Conv2d → BatchNorm2d → Activation

    This is the atom of Badger — almost every layer starts with this block.
    """
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, groups=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              autopad(kernel_size, padding), groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        # SiLU (Swish) is Badger's default — smooth, non-monotonic, works great
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Fused forward pass (for inference speedup after fusing BN into conv)."""
        return self.act(self.conv(x))


# =============================================================================
# 2. Bottleneck — residual learning block
# =============================================================================
class Bottleneck(nn.Module):
    """
    Standard bottleneck with optional residual connection.

    Structure: Conv(1x1, c//2) → Conv(3x3, c)  [+ shortcut]
    The 1x1 conv reduces channels, 3x3 conv restores them.
    """
    def __init__(self, in_channels, out_channels, shortcut=True, groups=1,
                 expansion=0.5):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.cv1 = Conv(in_channels, hidden_channels, 1, 1)
        self.cv2 = Conv(hidden_channels, out_channels, 3, 1, groups=groups)
        self.add = shortcut and in_channels == out_channels

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


# =============================================================================
# 3. C2f — YOLOv8's improved cross-stage partial fusion
# =============================================================================
class C2f(nn.Module):
    """
    C2f: CSP bottleneck with 2 convolutions + multiple Bottlenecks.

    This is Badger's improvement over YOLOv5's C3:
      - Splits input into two paths
      - Path 1: passes through N bottleneck blocks
      - Path 2: simple 1x1 conv
      - Concatenates both paths, then final 1x1 conv

    The "split and merge" design improves gradient flow and reduces computation.
    """
    def __init__(self, in_channels, out_channels, num_bottlenecks=1,
                 shortcut=False, groups=1, expansion=0.5):
        super().__init__()
        hidden_channels = int(out_channels * expansion)

        # Path 1: 1x1 conv to reduce channels
        self.cv1 = Conv(in_channels, 2 * hidden_channels, 1, 1)

        # Path 2: 1x1 conv (applied to second half of cv1 output + all bottleneck outputs)
        self.cv2 = Conv((2 + num_bottlenecks) * hidden_channels, out_channels, 1)

        # Bottleneck chain
        self.m = nn.ModuleList(
            Bottleneck(hidden_channels, hidden_channels, shortcut, groups, expansion=1.0)
            for _ in range(num_bottlenecks)
        )

    def forward(self, x):
        # Split cv1 output into two halves
        y = list(self.cv1(x).chunk(2, dim=1))
        # Each bottleneck takes the last output and appends
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, dim=1))


# =============================================================================
# 4. SPPF — Spatial Pyramid Pooling (Fast)
# =============================================================================
class SPPF(nn.Module):
    """
    Spatial Pyramid Pooling — Fast version.

    Instead of parallel max pools, SPPF uses sequential max pools.
    Same receptive field, fewer computations.

    Structure: Conv → [MaxPool → MaxPool → MaxPool] → Concat → Conv
    Each MaxPool is 5x5, so stacked they give 5x5, 9x9, 13x13 receptive fields.
    """
    def __init__(self, in_channels, out_channels, kernel_size=5):
        super().__init__()
        hidden_channels = in_channels // 2
        self.cv1 = Conv(in_channels, hidden_channels, 1, 1)
        self.cv2 = Conv(hidden_channels * 4, out_channels, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], dim=1))


# =============================================================================
# 5. Detect — the detection head
# =============================================================================
class DFL(nn.Module):
    """
    Distribution Focal Loss (DFL) module.

    Instead of regressing box coordinates directly, DFL predicts a discrete
    distribution over possible edge offsets. This gives finer localization,
    especially at object boundaries.

    reg_max=16 means we discretize each edge into 16 bins [0, 1, ..., 15].

    Math:
      predicted_edge = Σ_i softmax(logits_i) × i
    where i ∈ {0, 1, ..., reg_max-1} and logits are the raw predictions.

    This is a parameter-free weighted sum — no conv needed.
    """
    def __init__(self, reg_max=16):
        super().__init__()
        self.reg_max = reg_max

    def forward(self, x):
        """
        Args:
            x: [B, 4*reg_max, H, W] — raw logits for 4 edges × reg_max bins

        Returns:
            [B, 4, H, W] — precise edge offsets (weighted sum over bins)
        """
        b, c, h, w = x.shape
        # Reshape: [B, 4*reg_max, H, W] → [B, 4, reg_max, H*W]
        x = x.view(b, 4, self.reg_max, h * w)
        # Softmax over the distribution dimension: P(bin) for each edge
        x = x.softmax(dim=2)
        # Weighted sum: Σ P(bin_i) × i → expected bin value
        # Create bin indices [0, 1, ..., reg_max-1]
        bin_indices = torch.arange(self.reg_max, device=x.device, dtype=x.dtype)
        bin_indices = bin_indices.view(1, 1, -1, 1)  # [1, 1, reg_max, 1]
        x = (x * bin_indices).sum(dim=2)  # [B, 4, H*W]
        # Reshape back: [B, 4, H*W] → [B, 4, H, W]
        x = x.view(b, 4, h, w)
        return x


class Detect(nn.Module):
    """
    Badger Decoupled Detection Head.

    "Decoupled" means classification and regression are separate branches.
    This was found to improve both tasks compared to a shared head.

    For each FPN level (P3, P4, P5):
      Input → [cls_branch: Conv → Conv → Conv] → class scores
      Input → [reg_branch: Conv → Conv → Conv] → bbox regression + DFL
    """
    def __init__(self, num_classes=80, channels=None, reg_max=16):
        super().__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.num_outputs = num_classes  # One score per class
        self.channels = channels or [256, 512, 1024]  # Channels for P3, P4, P5

        # For each detection scale, create two branches
        self.cls_branches = nn.ModuleList()
        self.reg_branches = nn.ModuleList()

        for ch in self.channels:
            # Classification branch: 2 convs + output
            cls_branch = nn.Sequential(
                Conv(ch, ch, 3),
                Conv(ch, ch, 3),
                nn.Conv2d(ch, self.num_outputs, 1)
            )
            self.cls_branches.append(cls_branch)

            # Regression branch: 2 convs + bbox output (4 edges) + DFL
            reg_branch = nn.Sequential(
                Conv(ch, ch, 3),
                Conv(ch, ch, 3),
                nn.Conv2d(ch, 4 * reg_max, 1)
            )
            self.reg_branches.append(reg_branch)

        self.dfl = DFL(reg_max)

        # Initialize biases for stable training
        self._init_biases()

    def _init_biases(self):
        """Initialize detection biases so the model starts predicting reasonable values."""
        for cls_branch, reg_branch in zip(self.cls_branches, self.reg_branches):
            # Classification: start with small negative bias → low confidence
            nn.init.constant_(cls_branch[-1].bias, -4.0)
            # Regression: start with small positive bias → small boxes near center
            nn.init.constant_(reg_branch[-1].bias, 0.0)

    def forward(self, features):
        """
        Args:
            features: list of feature maps [P3, P4, P5]
                      shapes: [B, 256, 80, 80], [B, 512, 40, 40], [B, 1024, 20, 20]

        Returns:
            cls_scores: list of [B, num_classes, H, W] — per-pixel class probabilities
            bbox_preds: list of [B, 4, H, W] — per-pixel box predictions
        """
        cls_scores = []
        bbox_preds = []

        for i, feat in enumerate(features):
            cls_out = self.cls_branches[i](feat)
            reg_out = self.reg_branches[i](feat)

            # Apply DFL to get precise bbox coordinates
            bbox_out = self.dfl(reg_out)
            # bbox_out shape: [B, 4, H, W]

            cls_scores.append(cls_out)
            bbox_preds.append(bbox_out)

        return cls_scores, bbox_preds


# =============================================================================
# Utility: make layer from config
# =============================================================================
def make_divisible(x, divisor=8):
    """Round up to nearest multiple of divisor (good for GPU tensor cores)."""
    return int(math.ceil(x / divisor) * divisor)
