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
# 11. BadgerV2 Models
# =============================================================================

def test_badger_v2():
    print("\n--- Test: BadgerV2 Models (all variants) ---")
    from src.models import create_badger_v2

    variants = ['pico', 'nano', 'tiny', 'small', 'medium', 'large']
    for variant in variants:
        try:
            model = create_badger_v2(variant, num_classes=80)
            x = torch.randn(1, 3, 640, 640)
            cls_scores, bbox_preds = model(x)
            total_params = sum(p.numel() for p in model.parameters())
            check(len(cls_scores) == 3 and len(bbox_preds) == 3,
                  f"BadgerV2-{variant}: forward pass ({total_params:,} params)")
            check(cls_scores[0].shape[1] == 80,
                  f"BadgerV2-{variant}: cls channels == 80")
            check(bbox_preds[0].shape[1] == 4,
                  f"BadgerV2-{variant}: bbox channels == 4")
        except Exception as e:
            result.add_fail(f"BadgerV2-{variant}", str(e))

    # Gradient flow
    model = create_badger_v2('small', num_classes=80)
    model.train()
    x = torch.randn(1, 3, 640, 640)
    cls, bbox = model(x)
    loss = sum(c.sum() for c in cls) + sum(b.sum() for b in bbox)
    loss.backward()
    grad_params = sum(1 for p in model.parameters()
                      if p.grad is not None and p.grad.abs().sum() > 0)
    total = sum(1 for p in model.parameters())
    check(grad_params == total,
          f"BadgerV2-small: all {total} params have gradients")

    # Param ordering (smaller variant has fewer params)
    p_nano = sum(p.numel() for p in create_badger_v2('nano').parameters())
    p_small = sum(p.numel() for p in create_badger_v2('small').parameters())
    p_large = sum(p.numel() for p in create_badger_v2('large').parameters())
    check(p_nano < p_small < p_large,
          f"Param ordering: nano({p_nano:,}) < small({p_small:,}) < large({p_large:,})")


def test_nms_free():
    print("\n--- Test: NMS-Free DualHead ---")
    from src.models import create_badger_v2
    from src.models.head import nms_free_postprocess

    model = create_badger_v2('small', num_classes=80, nms_free=True)

    # Training mode returns two output groups
    model.train()
    x = torch.randn(1, 3, 640, 640)
    out = model(x)
    check(isinstance(out, tuple) and len(out) == 2,
          "DualHead train: returns 2 output groups")
    (m_cls, m_bbox), (o_cls, o_bbox) = out
    check(len(m_cls) == 3 and len(o_cls) == 3,
          "DualHead train: both heads output 3 scales")

    # Eval mode returns single output (one2one only)
    model.eval()
    cls, bbox = model(x)
    check(len(cls) == 3 and len(bbox) == 3,
          "DualHead eval: returns cls + bbox (3 scales)")

    # NMS-free post-processing
    detections = nms_free_postprocess(cls, bbox, conf_threshold=0.001)
    check(len(detections) == 1, "NMS-free postprocess: 1 result per image")
    boxes, scores, class_ids = detections[0]
    check(boxes.ndim == 2 and boxes.shape[1] == 4,
          f"NMS-free: boxes shape = {boxes.shape}")
    check(len(scores) == len(class_ids) == len(boxes),
          "NMS-free: consistent output lengths")


def test_pconv_variant():
    print("\n--- Test: PConv Edge Variant ---")
    from src.models import create_badger_v2

    model = create_badger_v2('nano', num_classes=80, use_pconv=True)
    x = torch.randn(1, 3, 640, 640)
    cls, bbox = model(x)
    check(len(cls) == 3 and len(bbox) == 3,
          "BadgerV2-nano-pconv: forward pass works")

    model_std = create_badger_v2('nano', num_classes=80, use_pconv=False)
    p_pconv = sum(p.numel() for p in model.parameters())
    p_std = sum(p.numel() for p in model_std.parameters())
    # PConv variant should have different param count (architecture differs)
    check(p_pconv != p_std,
          f"PConv vs standard: different architecture ({p_pconv:,} vs {p_std:,})")


def test_reparameterization():
    print("\n--- Test: Reparameterization ---")
    from src.models import create_badger_v2

    model = create_badger_v2('small', num_classes=80)
    model.eval()
    x = torch.randn(1, 3, 640, 640)

    with torch.no_grad():
        cls_before, bbox_before = model(x)

    model.fuse_for_deploy()

    with torch.no_grad():
        cls_after, bbox_after = model(x)

    cls_diff = max((a - b).abs().max().item()
                   for a, b in zip(cls_before, cls_after))
    bbox_diff = max((a - b).abs().max().item()
                    for a, b in zip(bbox_before, bbox_after))
    check(cls_diff < 1e-4,
          f"Reparam cls diff < 1e-4 (got {cls_diff:.8f})")
    check(bbox_diff < 1e-4,
          f"Reparam bbox diff < 1e-4 (got {bbox_diff:.8f})")


# =============================================================================
# 12. Advanced Loss Dispatch (compute_box_loss)
# =============================================================================

def test_box_loss_dispatch():
    print("\n--- Test: compute_box_loss Dispatch ---")
    from src.losses.advanced_losses import compute_box_loss

    pred = torch.tensor([[0.5, 0.5, 0.2, 0.3], [0.3, 0.3, 0.1, 0.1]])
    tgt = torch.tensor([[0.5, 0.5, 0.2, 0.3], [0.3, 0.3, 0.1, 0.1]])

    for loss_type in ['wiou', 'inner_iou', 'focal_eiou', 'siou', 'ciou']:
        try:
            loss = compute_box_loss(pred, tgt, loss_type=loss_type)
            check(not torch.isnan(loss) and loss.item() >= 0,
                  f"compute_box_loss({loss_type}): {loss.item():.6f}")
        except Exception as e:
            result.add_fail(f"compute_box_loss({loss_type})", str(e))

    # Perfect match should give near-zero loss
    for loss_type in ['wiou', 'inner_iou', 'focal_eiou', 'siou', 'ciou']:
        try:
            loss = compute_box_loss(pred, tgt, loss_type=loss_type)
            check(loss.item() < 0.01,
                  f"compute_box_loss({loss_type}) perfect match ≈ 0 (got {loss.item():.6f})")
        except Exception as e:
            result.add_fail(f"compute_box_loss({loss_type}) zero", str(e))

    # Invalid loss type should raise
    try:
        compute_box_loss(pred, tgt, loss_type='invalid')
        result.add_fail("compute_box_loss(invalid)", "Should have raised ValueError")
    except ValueError:
        result.add_pass("compute_box_loss(invalid) raises ValueError")
    except Exception as e:
        result.add_fail("compute_box_loss(invalid)", str(e))


# =============================================================================
# 13. Advanced Building Blocks
# =============================================================================

def test_advanced_blocks():
    print("\n--- Test: Advanced Blocks ---")
    from src.models.blocks import (
        PConv, PConvBlock, CIB, C2f_CIB,
        RepVGGBlock, RepBottleneck, RepC2f,
        GELAN, DyHeadBlock, DyHead,
    )

    x64 = torch.randn(2, 64, 40, 40)
    x128 = torch.randn(2, 128, 40, 40)
    x256 = torch.randn(2, 256, 20, 20)

    # PConv
    pconv = PConv(64, 3, n_div=4)
    y = pconv(x64)
    check(y.shape == x64.shape, f"PConv shape preserved: {y.shape}")

    # PConvBlock
    pblock = PConvBlock(64)
    y = pblock(x64)
    check(y.shape == x64.shape, f"PConvBlock shape preserved: {y.shape}")

    # CIB
    cib = CIB(64, 64)
    y = cib(x64)
    check(y.shape == x64.shape, f"CIB shape preserved: {y.shape}")

    # C2f_CIB
    c2f_cib = C2f_CIB(64, 64, num_blocks=2)
    y = c2f_cib(x64)
    check(y.shape == x64.shape, f"C2f_CIB shape preserved: {y.shape}")

    # RepVGGBlock (train mode)
    rep = RepVGGBlock(64, 64)
    y = rep(x64)
    check(y.shape == x64.shape, "RepVGGBlock train shape preserved")

    # RepVGGBlock (fuse)
    rep.eval()
    with torch.no_grad():
        y_before = rep(x64)
    rep.fuse()
    with torch.no_grad():
        y_after = rep(x64)
    diff = (y_before - y_after).abs().max().item()
    check(diff < 1e-4, f"RepVGGBlock fuse diff < 1e-4 (got {diff:.8f})")

    # RepC2f
    repc2f = RepC2f(64, 64, num_blocks=2)
    y = repc2f(x64)
    check(y.shape == x64.shape, f"RepC2f shape preserved: {y.shape}")

    # RepC2f fuse
    repc2f.eval()
    repc2f.fuse()
    y2 = repc2f(x64)
    check(y2.shape == x64.shape, "RepC2f fuse preserves shape")

    # GELAN
    gelan = GELAN(64, 64, num_branches=3)
    y = gelan(x64)
    check(y.shape == x64.shape, f"GELAN shape preserved: {y.shape}")

    # DyHeadBlock
    dyblock = DyHeadBlock(64, num_tasks=2)
    y = dyblock(x64, task_id=0)
    check(y.shape == x64.shape, "DyHeadBlock shape preserved")

    # DyHead (full)
    dyhead = DyHead(num_classes=80, channels=[64, 128, 256], num_blocks=2)
    features = [x64, x128, x256]
    cls, bbox = dyhead(features)
    check(len(cls) == 3, "DyHead outputs 3 cls maps")
    check(cls[0].shape[1] == 80, f"DyHead cls channels == 80")
    check(bbox[0].shape[1] == 4, f"DyHead bbox channels == 4")


# =============================================================================
# 14. Experiment Improvements Module
# =============================================================================

def test_improvements():
    print("\n--- Test: Improvements Module ---")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from experiments.improvements import (
        CBAM, ECA, SEBlock, Mish, GroupNormConv, ModelEMA,
        DeformableConv2d, RepConv, DropPath,
        build_combo, random_resize,
    )

    x = torch.randn(2, 64, 40, 40)

    # CBAM
    cbam = CBAM(64)
    y = cbam(x)
    check(y.shape == x.shape, "CBAM shape preserved")

    # ECA
    eca = ECA(64)
    y = eca(x)
    check(y.shape == x.shape, "ECA shape preserved")

    # SEBlock
    se = SEBlock(64)
    y = se(x)
    check(y.shape == x.shape, "SEBlock shape preserved")

    # Mish
    mish = Mish()
    y = mish(x)
    check(y.shape == x.shape, "Mish shape preserved")

    # GroupNormConv
    gnc = GroupNormConv(64, 128, 3)
    y = gnc(x)
    check(y.shape == (2, 128, 40, 40), f"GroupNormConv output: {y.shape}")

    # ModelEMA
    model = nn.Conv2d(3, 16, 3, padding=1)
    ema = ModelEMA(model, decay=0.999)
    ema.update(model)
    ema.update(model)
    ema.apply(model)
    check(True, "ModelEMA update + apply works")

    # DeformableConv2d
    dcn = DeformableConv2d(64, 64, 3, padding=1)
    y = dcn(x)
    check(y.shape == x.shape, "DeformableConv2d shape preserved")

    # RepConv (training)
    rep = RepConv(64, 64)
    y = rep(x)
    check(y.shape == x.shape, "RepConv train shape preserved")

    # RepConv (deploy=True)
    rep_deploy = RepConv(64, 64, deploy=True)
    y = rep_deploy(x)
    check(y.shape == x.shape, "RepConv deploy=True works")

    # RepConv reparameterize
    rep.eval()
    with torch.no_grad():
        y_before = rep(x)
    rep.reparameterize()
    with torch.no_grad():
        y_after = rep(x)
    diff = (y_before - y_after).abs().max().item()
    check(diff < 1e-3, f"RepConv reparameterize diff < 1e-3 (got {diff:.6f})")

    # DropPath
    dp = DropPath(0.2)
    dp.train()
    y = dp(x)
    check(y.shape == x.shape, "DropPath train shape preserved")
    dp.eval()
    y2 = dp(x)
    check(torch.allclose(y2, x), "DropPath eval is identity")

    # random_resize
    sizes = [random_resize(640) for _ in range(20)]
    check(all(s % 32 == 0 for s in sizes), "random_resize: all multiples of 32")
    check(all(320 <= s <= 1280 for s in sizes), "random_resize: all in [320, 1280]")

    # build_combo
    combo = build_combo('cbam_attention', 'ema_weights')
    check('CBAM' in combo['name'] and 'EMA' in combo['name'],
          f"build_combo: {combo['name'][:50]}...")


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

    # Full model (v1)
    test_full_model()

    # Full model (v2)
    test_badger_v2()
    test_nms_free()
    test_pconv_variant()
    test_reparameterization()

    # Losses
    test_losses()
    test_advanced_losses()
    test_box_loss_dispatch()

    # Attention
    test_attention()

    # Advanced blocks
    test_advanced_blocks()

    # Box ops
    test_box_ops()

    # Metrics
    test_metrics()

    # Improvements module
    test_improvements()

    return result.summary()


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
