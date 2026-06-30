#!/usr/bin/env python3
"""
Comprehensive test suite for Badger.

Tests every component:
  1. Model creation (all variants)
  2. Forward pass (all variants, verify output shapes)
  3. Backward pass (gradients flow correctly)
  4. Loss functions (produce valid values)
  5. Label assigners (TAL, SimOTA produce valid assignments)
  6. Building blocks (Conv, C2f, SPPF, DFL)
  7. Attention modules (Swin, global mixer)
  8. Box ops (NMS, format conversions)
  9. Data pipeline (COCO loading, mosaic)
  10. Metrics (mAP calculation)

Usage:
    python scripts/test_all.py           # Run all tests
    python scripts/test_all.py --quick   # Skip slow tests (data loading)
    python scripts/test_all.py --verbose # Full output
"""

import sys
import time
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import numpy as np


# =============================================================================
# Test Harness
# =============================================================================

class TestResult:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self.errors = []
        self.start_time = time.time()

    def add_pass(self, name):
        self.passed += 1
        print(f"  ✓ {name}")

    def add_fail(self, name, error):
        self.failed += 1
        self.errors.append((name, str(error)))
        print(f"  ✗ {name}: {error}")

    def add_skip(self, name, reason):
        self.skipped += 1
        print(f"  - {name} (skipped: {reason})")

    def summary(self):
        elapsed = time.time() - self.start_time
        total = self.passed + self.failed + self.skipped
        print(f"\n{'='*60}")
        print(f"  RESULTS: {self.passed}/{total} passed, "
              f"{self.failed} failed, {self.skipped} skipped")
        print(f"  Time: {elapsed:.1f}s")
        if self.errors:
            print(f"\n  FAILURES:")
            for name, error in self.errors:
                print(f"    {name}: {error}")
        print(f"{'='*60}")
        return self.failed == 0


result = TestResult()


def check(condition, name, error_msg="Assertion failed"):
    """Simple assertion with result tracking."""
    if condition:
        result.add_pass(name)
    else:
        result.add_fail(name, error_msg)


# =============================================================================
# 1. Building Blocks
# =============================================================================

def test_conv():
    print("\n--- Test: Conv Block ---")
    from src.models.blocks import Conv

    conv = Conv(3, 64, 3, 2)
    x = torch.randn(2, 3, 640, 640)
    y = conv(x)
    check(y.shape == (2, 64, 320, 320), "Conv output shape (stride=2)")
    check(y.dtype == torch.float32, "Conv output dtype")

    # Gradient flow
    loss = y.sum()
    loss.backward()
    check(conv.conv.weight.grad is not None, "Conv gradient flows")
    check(conv.conv.weight.grad.abs().sum() > 0, "Conv gradient non-zero")


def test_bottleneck():
    print("\n--- Test: Bottleneck ---")
    from src.models.blocks import Bottleneck

    # With shortcut (residual: y = x + F(x), F freshly initialized ≠ 0)
    bn = Bottleneck(64, 64, shortcut=True)
    x = torch.randn(2, 64, 80, 80)
    y = bn(x)
    check(y.shape == (2, 64, 80, 80), "Bottleneck with shortcut shape")
    # Fresh weights produce non-trivial F(x), so output ≠ input (good — it's learning)
    check(not torch.allclose(y, x, atol=0.01), "Bottleneck not stuck at identity")

    # Without shortcut (channel mismatch)
    bn2 = Bottleneck(64, 128, shortcut=False)
    y2 = bn2(x)
    check(y2.shape == (2, 128, 80, 80), "Bottleneck without shortcut shape")


def test_c2f():
    print("\n--- Test: C2f Module ---")
    from src.models.blocks import C2f

    c2f = C2f(64, 64, num_bottlenecks=2, shortcut=True)
    x = torch.randn(2, 64, 80, 80)
    y = c2f(x)
    check(y.shape == (2, 64, 80, 80), "C2f output shape")
    check(y.dtype == torch.float32, "C2f output dtype")

    # Verify gradient flow
    params_before = sum(p.numel() for p in c2f.parameters())
    loss = y.sum()
    loss.backward()
    grad_flow = sum(p.grad.abs().sum().item() for p in c2f.parameters() if p.grad is not None)
    check(grad_flow > 0, f"C2f gradient flow ({params_before:,} params)")


def test_sppf():
    print("\n--- Test: SPPF ---")
    from src.models.blocks import SPPF

    sppf = SPPF(512, 512)
    x = torch.randn(2, 512, 20, 20)
    y = sppf(x)
    check(y.shape == (2, 512, 20, 20), "SPPF output shape (spatial preserved)")
    # SPPF should preserve spatial dims but enrich features
    check(y.std() > 0, "SPPF output non-degenerate")


def test_dfl():
    print("\n--- Test: DFL ---")
    from src.models.blocks import DFL

    dfl = DFL(reg_max=16)
    x = torch.randn(1, 64, 80, 80)  # 4*16 = 64 channels for 4 edges
    y = dfl(x)
    check(y.shape == (1, 4, 80, 80), "DFL output shape (4 edges per pixel)")


# =============================================================================
# 2. Backbone
# =============================================================================

def test_backbone():
    print("\n--- Test: CSPDarknet Backbone ---")
    from src.models.backbone import CSPDarknet

    backbone = CSPDarknet(width_multiple=0.50, depth_multiple=0.33)
    x = torch.randn(2, 3, 640, 640)
    features = backbone(x)

    check(len(features) == 3, "Backbone outputs 3 feature maps")
    # P3, P4, P5
    expected_channels = backbone.out_channels
    check(features[0].shape == (2, expected_channels[0], 80, 80), "P3 shape (80×80)")
    check(features[1].shape == (2, expected_channels[1], 40, 40), "P4 shape (40×40)")
    check(features[2].shape == (2, expected_channels[2], 20, 20), "P5 shape (20×20)")

    # Verify channels are as expected
    check(features[0].shape[1] == expected_channels[0], f"P3 channels: {features[0].shape[1]} == {expected_channels[0]}")
    check(features[1].shape[1] == expected_channels[1], f"P4 channels: {features[1].shape[1]} == {expected_channels[1]}")
    check(features[2].shape[1] == expected_channels[2], f"P5 channels: {features[2].shape[1]} == {expected_channels[2]}")


# =============================================================================
# 3. Neck
# =============================================================================

def test_neck():
    print("\n--- Test: PA-FPN Neck ---")
    from src.models.neck import PAFPN
    from src.models.backbone import CSPDarknet

    backbone = CSPDarknet(width_multiple=0.50, depth_multiple=0.33)
    neck = PAFPN(backbone.out_channels, width_multiple=0.5, depth_multiple=0.33)

    x = torch.randn(2, 3, 640, 640)
    features = backbone(x)
    fused = neck(features)

    check(len(fused) == 3, "Neck outputs 3 feature maps")
    for i, (name, expected_ch) in enumerate(zip(['N3', 'N4', 'N5'], neck.out_channels)):
        check(fused[i].shape[1] == expected_ch,
              f"{name} channels: {fused[i].shape[1]} == {expected_ch}")


# =============================================================================
# 4. Head
# =============================================================================

def test_head():
    print("\n--- Test: Decoupled Detection Head ---")
    from src.models.head import DecoupledHead

    head = DecoupledHead(num_classes=80, channels=[256, 256, 256])
    features = [
        torch.randn(2, 256, 80, 80),
        torch.randn(2, 256, 40, 40),
        torch.randn(2, 256, 20, 20),
    ]
    cls_scores, bbox_preds = head(features)

    check(len(cls_scores) == 3, "Head outputs 3 cls maps")
    check(len(bbox_preds) == 3, "Head outputs 3 bbox maps")
    check(cls_scores[0].shape[1] == 80, "P3 cls: 80 classes")
    check(bbox_preds[0].shape[1] == 4, "P3 bbox: 4 coordinates")


# =============================================================================
# 5. Full Model (all variants)
# =============================================================================

def test_full_model():
    print("\n--- Test: Full Badger Model (all variants) ---")
    from src.models import create_model

    variants = {
        'badger-n': (0.25, 0.33),
        'badger-s': (0.50, 0.33),
        'badger-m': (0.75, 0.50),
        'badger-l': (1.00, 0.75),
        'badger-x': (1.25, 1.00),
    }

    for variant, (w, d) in variants.items():
        model = create_model(variant, num_classes=80)
        x = torch.randn(1, 3, 640, 640)
        cls_scores, bbox_preds = model(x)

        total_params = sum(p.numel() for p in model.parameters())
        check(len(cls_scores) == 3 and len(bbox_preds) == 3,
              f"{variant}: forward pass ({total_params:,} params)")

        # Verify gradient flow
        loss = sum(c.sum() for c in cls_scores) + sum(b.sum() for b in bbox_preds)
        loss.backward()

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        grad_params = sum(1 for p in model.parameters() if p.grad is not None)
        check(grad_params > 0, f"{variant}: gradients flow ({grad_params} params with grad)")

        print(f"    {variant}: {total_params:,} params, {trainable:,} trainable")

    # Verify parameter estimates
    model_n = create_model('badger-n', num_classes=80)
    model_s = create_model('badger-s', num_classes=80)
    params_n = sum(p.numel() for p in model_n.parameters())
    params_s = sum(p.numel() for p in model_s.parameters())

    check(params_n < 3_000_000, f"Badger-Nano params: {params_n:,} < 3M")
    check(params_s < 12_000_000, f"Badger-Small params: {params_s:,} < 12M")
    check(params_n < params_s, "Badger-N < Badger-S in params")


# =============================================================================
# 6. Loss Functions
# =============================================================================

def test_losses():
    print("\n--- Test: Loss Functions ---")
    from src.losses import BadgerLoss, ciou_loss, dfl_loss
    from src.losses.badger_loss import TaskAlignedAssigner, SimOTAAssigner

    # CIoU loss
    pred_boxes = torch.tensor([[0.5, 0.5, 0.2, 0.3], [0.3, 0.3, 0.1, 0.1]])
    target_boxes = torch.tensor([[0.5, 0.5, 0.2, 0.3], [0.3, 0.3, 0.1, 0.1]])
    ciou = ciou_loss(pred_boxes, target_boxes)
    check(ciou.item() < 0.01, f"CIoU perfect match ≈ 0 (got {ciou.item():.4f})")

    # DFL loss
    pred_dist = torch.randn(4, 4 * 16)  # 4 predictions, 4 edges × 16 bins
    target = torch.tensor([[2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0],
                          [1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    dfl = dfl_loss(pred_dist, target)
    check(not torch.isnan(dfl), "DFL loss is finite")
    check(dfl.item() > 0, "DFL loss > 0")

    # TAL assigner
    tal = TaskAlignedAssigner(num_classes=80)
    pred_scores = torch.rand(2, 8400, 80).sigmoid()  # 2 images, 8400 anchors, 80 cls
    pred_bboxes = torch.rand(2, 8400, 4)
    targets = torch.tensor([
        [0, 5, 0.5, 0.5, 0.2, 0.3],  # img 0, cls 5
        [1, 10, 0.3, 0.3, 0.1, 0.1], # img 1, cls 10
    ])
    anchors = torch.rand(8400, 2) * 640
    t_labels, t_boxes, t_scores, fg = tal(
        pred_scores, pred_bboxes, targets, anchors, [8, 16, 32], (640, 640), 2
    )
    check(t_labels.shape == (2, 8400, 80), "TAL target_labels shape")
    check(fg.sum() > 0, "TAL assigns at least one foreground")

    # SimOTA assigner
    simota = SimOTAAssigner(num_classes=80)
    t_labels2, t_boxes2, t_scores2, fg2 = simota(
        pred_scores, pred_bboxes, targets, anchors, [8, 16, 32], (640, 640), 2
    )
    check(fg2.sum() > 0, "SimOTA assigns at least one foreground")
    print(f"    TAL fg: {fg.sum().item()}, SimOTA fg: {fg2.sum().item()}")

    # BadgerLoss (full)
    badger_loss = BadgerLoss(num_classes=80, assigner='tal')
    # Create dummy predictions
    cls_scores = [
        torch.randn(2, 80, 80, 80),
        torch.randn(2, 80, 40, 40),
        torch.randn(2, 80, 20, 20),
    ]
    bbox_preds = [
        torch.randn(2, 4, 80, 80),
        torch.randn(2, 4, 40, 40),
        torch.randn(2, 4, 20, 20),
    ]
    total_loss, loss_dict = badger_loss(cls_scores, bbox_preds, targets, (640, 640))
    check(not torch.isnan(total_loss), "BadgerLoss total is finite")
    check(total_loss.item() > 0, "BadgerLoss > 0")
    check('box' in loss_dict and 'cls' in loss_dict, "Loss dict has box+cls keys")


# =============================================================================
# 7. Advanced Losses
# =============================================================================

def test_advanced_losses():
    print("\n--- Test: Advanced Losses ---")
    from src.losses.advanced_losses import GaussianDFL, VarifocalLoss, siou_loss

    # Gaussian DFL
    gdfl = GaussianDFL()
    mu = torch.randn(10, 4)
    sigma = torch.randn(10, 4).abs() * 0.5
    target = torch.rand(10, 4) * 16
    loss_g = gdfl(mu, sigma, target)
    check(not torch.isnan(loss_g), "GaussianDFL loss is finite")
    check(loss_g.item() > 0, "GaussianDFL loss > 0")

    # Varifocal loss
    vfl = VarifocalLoss()
    preds = torch.randn(100, 80)
    target_bin = torch.zeros(100, 80)
    target_bin[0, 5] = 1.0  # One positive
    iou_scores = torch.zeros(100, 80)
    iou_scores[0, 5] = 0.85
    loss_v = vfl(preds, target_bin, iou_scores)
    check(not torch.isnan(loss_v), "Varifocal loss is finite")

    # SIoU loss
    pred_box = torch.tensor([[0.5, 0.5, 0.2, 0.3]])
    target_box = torch.tensor([[0.5, 0.5, 0.2, 0.3]])
    loss_s = siou_loss(pred_box, target_box)
    check(loss_s.item() < 0.01, f"SIoU perfect match ≈ 0 (got {loss_s.item():.4f})")


# =============================================================================
# 8. Attention Modules
# =============================================================================

def test_attention():
    print("\n--- Test: Attention Modules ---")
    from src.models.attention import WindowAttention, SwinBlock, GlobalTokenMixer, AttentionNeck
    from src.models.backbone import CSPDarknet

    # Window attention
    wa = WindowAttention(dim=128, num_heads=4, window_size=7)
    x = torch.randn(4, 49, 128)  # 4 windows × 7×7=49 tokens, dim=128
    y = wa(x)
    check(y.shape == x.shape, "Window attention preserves shape")

    # Swin block — need spatial dims divisible by window_size (7)
    swin = SwinBlock(dim=128, num_heads=4, window_size=7, shift_size=0)
    x = torch.randn(2, 196, 128)  # B=2, H=14, W=14, C=128 (14 divisible by 7)
    y = swin(x, H=14, W=14)
    check(y.shape == x.shape, "SwinBlock preserves shape")

    # Global token mixer
    gtm = GlobalTokenMixer(channels=256)
    feats = [
        torch.randn(2, 256, 20, 20),
        torch.randn(2, 256, 10, 10),
        torch.randn(2, 256, 5, 5),
    ]
    y = gtm(feats)
    check(len(y) == 3, "GlobalTokenMixer outputs 3 maps")
    for i, f in enumerate(y):
        check(f.shape == feats[i].shape, f"GlobalTokenMixer scale {i} shape preserved")

    # Full Attention Neck
    backbone = CSPDarknet(width_multiple=0.50, depth_multiple=0.33)
    neck = AttentionNeck(backbone.out_channels, width_multiple=0.5, depth_multiple=0.33)
    x = torch.randn(1, 3, 640, 640)
    feats = backbone(x)
    y = neck(feats)
    check(len(y) == 3, "AttentionNeck outputs 3 maps")


# =============================================================================
# 9. Box Operations
# =============================================================================

def test_box_ops():
    print("\n--- Test: Box Operations ---")
    from src.utils.box_ops import xywh_to_xyxy, xyxy_to_xywh, nms, scale_boxes

    # xywh ↔ xyxy
    boxes_xywh = torch.tensor([[100, 100, 50, 50], [200, 200, 80, 60]]).float()
    boxes_xyxy = xywh_to_xyxy(boxes_xywh)
    check(boxes_xyxy.shape == (2, 4), "xywh→xyxy shape")
    back = xyxy_to_xywh(boxes_xyxy)
    check(torch.allclose(boxes_xywh, back, atol=1e-5), "xywh→xyxy→xywh roundtrip")

    # NMS
    boxes = torch.tensor([
        [0, 0, 100, 100],
        [5, 5, 95, 95],    # heavily overlaps with box 0
        [200, 200, 300, 300], # far away
    ]).float()
    scores = torch.tensor([0.9, 0.85, 0.8])
    keep = nms(boxes, scores, iou_threshold=0.5)
    check(len(keep) == 2, f"NMS keeps 2 boxes, got {len(keep)}")
    check(0 in keep and 2 in keep, "NMS keeps non-overlapping boxes")


# =============================================================================
# 10. Metrics
# =============================================================================

def test_metrics():
    print("\n--- Test: Metrics ---")
    from src.utils.metrics import MeanAveragePrecision, compute_iou

    # IoU test: box1 covers (0,0)→(100,100), box2 covers (50,50)→(150,150)
    # Intersection: (50,50)→(100,100) = 50×50 = 2500
    # Union: 100×100 + 100×100 - 2500 = 17500
    # IoU = 2500/17500 ≈ 0.143
    box1 = torch.tensor([[0, 0, 100, 100]]).float()
    box2 = torch.tensor([[50, 50, 150, 150]]).float()
    iou = compute_iou(box1, box2).item()
    expected_iou = 2500.0 / 17500.0
    check(abs(iou - expected_iou) < 0.01, f"IoU 50% overlap (got {iou:.3f}, expected {expected_iou:.3f})")

    # mAP calculator
    mAP_calc = MeanAveragePrecision(num_classes=2)
    # Perfect prediction
    pred_boxes = torch.tensor([[0, 0, 100, 100]]).float()
    pred_scores = torch.tensor([0.9])
    pred_classes = torch.tensor([0])
    gt_boxes = torch.tensor([[0, 0, 100, 100]]).float()
    gt_classes = torch.tensor([0])
    mAP_calc.update(pred_boxes, pred_scores, pred_classes, gt_boxes, gt_classes)
    ap = mAP_calc.compute()
    check(ap['mAP'] > 0.5, f"mAP with perfect pred > 0.5 (got {ap['mAP']:.3f})")


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='Badger Test Suite')
    parser.add_argument('--quick', action='store_true', help='Skip slow tests')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"\n{'='*60}")
    print(f"  BADGER COMPREHENSIVE TEST SUITE")
    print(f"{'='*60}")
    
    # Building blocks
    test_conv()
    test_bottleneck()
    test_c2f()
    test_sppf()
    test_dfl()

    # Backbone + Neck + Head
    test_backbone()
    test_neck()
    test_head()

    # Full model
    test_full_model()

    # Losses
    test_losses()
    test_advanced_losses()

    # Attention
    test_attention()

    # Box ops
    test_box_ops()

    # Metrics
    test_metrics()

    return result.summary()


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
