"""
Detection heads for Badger.

The head takes fused features from the neck and produces:
  1. Classification scores — "what object is this?"
  2. Bounding box coordinates — "where is it?"
"""

import torch
import torch.nn as nn
from .blocks import Conv, DFL


class DecoupledHead(nn.Module):
    """
    Decoupled detection head — Badger's default.

    "Decoupled" means classification and regression use separate convolutional
    branches. This is better than a shared head because:
      - Classification needs to know WHAT (semantic features)
      - Regression needs to know WHERE (spatial features)
    These tasks benefit from different feature representations.
    """
    def __init__(self, num_classes=80, channels=None, reg_max=16):
        super().__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.num_outputs = num_classes
        self.channels = channels or [256, 256, 256]  # Neck outputs same channel count

        self.cls_branches = nn.ModuleList()
        self.reg_branches = nn.ModuleList()

        for ch in self.channels:
            # Classification branch: two 3x3 convs + 1x1 output
            cls_branch = nn.Sequential(
                Conv(ch, ch, 3),
                Conv(ch, ch, 3),
                nn.Conv2d(ch, self.num_outputs, 1)
            )
            self.cls_branches.append(cls_branch)

            # Regression branch: two 3x3 convs + 1x1 output (4 * reg_max)
            # 4 = (left, top, right, bottom), reg_max = bins per edge
            reg_branch = nn.Sequential(
                Conv(ch, ch, 3),
                Conv(ch, ch, 3),
                nn.Conv2d(ch, 4 * reg_max, 1)
            )
            self.reg_branches.append(reg_branch)

        self.dfl = DFL(reg_max)

        self._init_biases()

    def _init_biases(self):
        """Initialize with small negative bias → models start conservative."""
        for cls_branch, reg_branch in zip(self.cls_branches, self.reg_branches):
            nn.init.constant_(cls_branch[-1].bias, -4.0)
            nn.init.constant_(reg_branch[-1].bias, 0.0)

    def forward(self, features):
        """
        Args:
            features: [N3, N4, N5] from neck
                      Shapes: [B, C, 80, 80], [B, C, 40, 40], [B, C, 20, 20]

        Returns:
            cls_scores: list of [B, num_classes, H, W]
            bbox_preds: list of [B, 4, H, W]
        """
        cls_scores = []
        bbox_preds = []

        for i, feat in enumerate(features):
            cls_out = self.cls_branches[i](feat)
            reg_out = self.reg_branches[i](feat)
            bbox_out = self.dfl(reg_out)
            cls_scores.append(cls_out)
            bbox_preds.append(bbox_out)

        return cls_scores, bbox_preds


class CoupledHead(nn.Module):
    """
    Coupled (shared) detection head — YOLOv5 style.

    Uses a single branch to predict both class and box.
    Simpler and faster, but generally less accurate than decoupled.

    Improvement experiment: compare decoupled vs. coupled to measure the
    accuracy/speed tradeoff for your specific use case.
    """
    def __init__(self, num_classes=80, channels=None):
        super().__init__()
        self.num_classes = num_classes
        self.num_outputs = num_classes + 5  # classes + (x, y, w, h, obj)
        self.channels = channels or [256, 256, 256]

        self.branches = nn.ModuleList()
        for ch in self.channels:
            branch = nn.Sequential(
                Conv(ch, ch, 3),
                nn.Conv2d(ch, self.num_outputs, 1)
            )
            self.branches.append(branch)

    def forward(self, features):
        outputs = []
        for i, feat in enumerate(features):
            outputs.append(self.branches[i](feat))
        return outputs
