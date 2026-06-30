"""
Full Badger model — assembles backbone, neck, and head.

This is the complete model you train end-to-end. It takes an image tensor
and outputs raw predictions (class scores + bounding boxes) at three scales.
"""

import torch
import torch.nn as nn
from .backbone import CSPDarknet
from .neck import PAFPN
from .head import DecoupledHead


class Badger(nn.Module):
    """
    Badger object detection model — inspired by YOLOX + YOLOv8.

    Pipeline:
      Image [B, 3, 640, 640]
        ↓
      Backbone (CSPDarknet) → [P3, P4, P5]  multi-scale features
        ↓
      Neck (PA-FPN)         → [N3, N4, N5]  fused features
        ↓
      Head (Decoupled)      → [cls_scores], [bbox_preds]  raw predictions
        ↓
      Post-processing       → [boxes, scores, class_ids]  final detections
    """

    def __init__(self, num_classes=80, width_multiple=0.50, depth_multiple=0.33):
        """
        Args:
            num_classes: number of object classes (COCO=80, VOC=20)
            width_multiple: channel scaling (0.25=nano, 0.50=small, 1.0=large)
            depth_multiple: layer scaling (0.33=nano/small, 1.0=large)

        Common variants:
            Badger-Nano:  width=0.25, depth=0.33  (~3M params)
            Badger-Small: width=0.50, depth=0.33  (~11M params)
            Badger-Medium: width=0.75, depth=0.67  (~26M params)
            Badger-Large: width=1.00, depth=1.00  (~44M params)
            Badger-XL:    width=1.25, depth=1.00  (~68M params)
        """
        super().__init__()

        # 1. Backbone — extracts features at multiple scales
        self.backbone = CSPDarknet(
            width_multiple=width_multiple,
            depth_multiple=depth_multiple
        )

        # 2. Neck — fuses features across scales
        self.neck = PAFPN(
            in_channels=self.backbone.out_channels,
            width_multiple=width_multiple,
            depth_multiple=depth_multiple
        )

        # 3. Head — produces detections
        self.head = DecoupledHead(
            num_classes=num_classes,
            channels=self.neck.out_channels
        )

        self.num_classes = num_classes
        self._strides = [8, 16, 32]  # Downsampling factors for P3, P4, P5

    def forward(self, x):
        """
        Args:
            x: image tensor [B, 3, H, W], normalized to [0, 1]

        Returns:
            cls_scores: list of [B, num_classes, H/8, W/8], [H/16, W/16], [H/32, W/32]
            bbox_preds: list of [B, 4, H/8, W/8], [H/16, W/16], [H/32, W/32]
        """
        features = self.backbone(x)
        fused_features = self.neck(features)
        cls_scores, bbox_preds = self.head(fused_features)
        return cls_scores, bbox_preds

    def get_strides(self):
        """Return stride for each detection scale (pixels in input per pixel in feature)."""
        return self._strides

    def count_parameters(self):
        """Return total and trainable parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


def create_model(variant='badger-s', num_classes=80, pretrained=False):
    """
    Factory function to create Badger models.

    Args:
        variant: 'badger-n', 'badger-s', 'badger-m', 'badger-l', 'badger-x'
        num_classes: number of classes for your dataset
        pretrained: load pretrained weights

    Returns:
        Badger model instance
    """
    variants = {
        'badger-n': (0.25, 0.33),
        'badger-s': (0.50, 0.33),
        'badger-m': (0.75, 0.50),
        'badger-l': (1.00, 0.75),
        'badger-x': (1.25, 1.00),
    }

    if variant not in variants:
        raise ValueError(f"Unknown variant '{variant}'. Options: {list(variants.keys())}")

    width, depth = variants[variant]
    model = Badger(num_classes=num_classes, width_multiple=width, depth_multiple=depth)

    if pretrained:
        # TODO: Load pretrained weights
        print(f"Warning: pretrained weights not yet implemented. Training from scratch.")

    return model
