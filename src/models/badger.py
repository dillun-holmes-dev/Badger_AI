"""
🦡 Badger — The unified object detection model.

Covers BOTH classic Badger and SOTA BadgerV2 in ONE file.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .blocks import (Conv, SPPF, C2f, DFL, make_divisible,
                      PConv, PConvBlock, RepVGGBlock, C2f_CIB, RepC2f,
                      RepBottleneck, AreaAttention, A2Block, RELAN,
                      fuse_model_reparam)
from .backbone import CSPDarknet
from .neck import PAFPN
from .head import DecoupledHead, DecoupledHeadWithQuality, QualityDecoupledHead


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

    def __init__(self, num_classes=80, width_multiple=0.50, depth_multiple=0.33,
                 head_type='decoupled', quality_exp=1.0,
                 use_p2=False, use_csprep=False, drop_path_rate=0.0):
        """
        Args:
            num_classes: number of object classes (COCO=80, VOC=20)
            width_multiple: channel scaling (0.25=nano, 0.50=small, 1.0=large)
            depth_multiple: layer scaling (0.33=nano/small, 1.0=large)
            head_type: 'decoupled' | 'quality_decoupled' | 'quality_gn'
            quality_exp: exponent for quality gating
            use_p2: enable P2 output (stride=4) for better small object detection
            use_csprep: use CSPRepLayer (RepVGG) in neck instead of C2f
            drop_path_rate: stochastic depth rate for regularization (0.0=off)

        Common variants:
            Badger-Nano:  width=0.25, depth=0.33  (~1.9M params)
            Badger-Small: width=0.50, depth=0.33  (~11M params)
            Badger-Medium: width=0.75, depth=0.67  (~26M params)
            Badger-Large: width=1.00, depth=1.00  (~44M params)
            Badger-XL:    width=1.25, depth=1.00  (~68M params)
        """
        super().__init__()

        # 1. Backbone — extracts features at multiple scales
        self.backbone = CSPDarknet(
            width_multiple=width_multiple,
            depth_multiple=depth_multiple,
            use_p2=use_p2,
            drop_path_rate=drop_path_rate,
        )

        # 2. Neck — fuses features across scales
        self.neck = PAFPN(
            in_channels=self.backbone.out_channels,
            width_multiple=width_multiple,
            depth_multiple=depth_multiple,
            use_csprep=use_csprep,
        )

        # 3. Head — produces detections
        self.head_type = head_type
        if head_type == 'quality_gn':
            self.head = QualityDecoupledHead(
                num_classes=num_classes,
                channels=self.neck.out_channels,
                quality_exp=quality_exp,
            )
        elif head_type == 'quality_decoupled':
            self.head = DecoupledHeadWithQuality(
                num_classes=num_classes,
                channels=self.neck.out_channels,
                quality_exp=quality_exp,
            )
        else:
            self.head = DecoupledHead(
                num_classes=num_classes,
                channels=self.neck.out_channels
            )

        self.num_classes = num_classes
        self.use_p2 = use_p2
        self._strides = [4, 8, 16, 32] if use_p2 else [8, 16, 32]

    def forward(self, x, return_raw_reg=False):
        """
        Args:
            x: image tensor [B, 3, H, W], normalized to [0, 1]
            return_raw_reg: if True, also return raw regression outputs
                           (before DFL softmax) for DFL loss computation

        Returns:
            cls_scores: list of [B, num_classes, H_i, W_i]
            bbox_preds: list of [B, 4, H_i, W_i]
            raw_reg: list of [B, 4*reg_max, H_i, W_i] or None

        Side effect:
            self._last_quality_scores is set to the quality head output
            (list of [B, 1, H_i, W_i]) when head_type is 'quality_decoupled'
            or 'quality_gn', or None for standard head. Access this after
            forward() to pass quality scores to the loss function.
        """
        features = self.backbone(x)
        fused_features = self.neck(features)

        if self.head_type in ('quality_decoupled', 'quality_gn'):
            if return_raw_reg:
                cls, bbox, quality, raw_reg = self.head(fused_features, return_raw_reg=True)
                self._last_quality_scores = quality
                return cls, bbox, raw_reg
            else:
                cls, bbox, quality = self.head(fused_features, return_raw_reg=False)
                self._last_quality_scores = quality
                return cls, bbox
        else:
            self._last_quality_scores = None
            return self.head(fused_features, return_raw_reg=return_raw_reg)

    def predict(self, x, conf_threshold=0.25, max_det=300):
        """
        Run inference with quality-aware post-processing.

        Args:
            x: image tensor [B, 3, H, W]
            conf_threshold: minimum confidence score
            max_det: maximum detections per image

        Returns:
            list of (boxes, scores, class_ids) per image
        """
        features = self.backbone(x)
        fused_features = self.neck(features)

        if self.head_type in ('quality_decoupled', 'quality_gn'):
            from .head import quality_aware_postprocess
            cls_scores, bbox_preds, quality_scores = self.head(fused_features)
            return quality_aware_postprocess(
                cls_scores, bbox_preds, quality_scores,
                conf_threshold=conf_threshold, max_det=max_det,
                quality_exp=self.head.get_quality_exp()
            )
        else:
            from .head import nms_free_postprocess
            cls_scores, bbox_preds = self.head(fused_features)
            return nms_free_postprocess(
                cls_scores, bbox_preds,
                conf_threshold=conf_threshold, max_det=max_det
            )

    def get_strides(self):
        """Return stride for each detection scale (pixels in input per pixel in feature)."""
        return self._strides

    def count_parameters(self):
        """Return total and trainable parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


def create_model(variant='badger-s', num_classes=80, pretrained=False,
                  head_type='decoupled', quality_exp=1.0,
                  use_p2=False, use_csprep=False, drop_path_rate=0.0):
    """
    Factory function to create Badger models.

    Args:
        variant: 'badger-n', 'badger-s', 'badger-m', 'badger-l', 'badger-x'
        num_classes: number of classes for your dataset
        pretrained: load pretrained weights
        head_type: 'decoupled' | 'quality_decoupled' | 'quality_gn'
        quality_exp: exponent for quality gating
        use_p2: enable P2 level for better small object detection
        use_csprep: use CSPRepLayer (RepVGG) in neck — reparameterizable
        drop_path_rate: stochastic depth rate (0.0=off, 0.1=mild, 0.2=strong)

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
    model = Badger(num_classes=num_classes, width_multiple=width, depth_multiple=depth,
                   head_type=head_type, quality_exp=quality_exp,
                   use_p2=use_p2, use_csprep=use_csprep,
                   drop_path_rate=drop_path_rate)

    if pretrained:
        # TODO: Load pretrained weights
        print(f"Warning: pretrained weights not yet implemented. Training from scratch.")

    return model


# =============================================================================
# Badger SOTA — Next-gen (PConv/RepC2f + BiFPN + DualHead)
# =============================================================================

# =============================================================================
# 1. Backbone v2 — PConv + RepC2f + C2f_CIB
# =============================================================================

class BackboneV2(nn.Module):
    """
    Modernized backbone combining:
      - RepC2f in early stages (reparameterizable for free accuracy)
      - C2f_CIB in later stages (compact inverted bottleneck for efficiency)
      - PConvBlock as drop-in efficient attention alternative
      - SPPF at the end for multi-scale receptive field

    Scaling is controlled by width_multiple and depth_multiple:
      - width: scales channel counts
      - depth: scales number of blocks per stage
    """

    def __init__(self, width_multiple=0.5, depth_multiple=0.33,
                 use_pconv=False, act='silu'):
        super().__init__()
        self.width_multiple = width_multiple
        self.depth_multiple = depth_multiple

        # Channel widths (base values, scaled by width_multiple)
        c1 = make_divisible(64 * width_multiple, 8)
        c2 = make_divisible(128 * width_multiple, 8)
        c3 = make_divisible(256 * width_multiple, 8)
        c4 = make_divisible(512 * width_multiple, 8)
        c5 = make_divisible(1024 * width_multiple, 8)

        # Block counts (scaled by depth_multiple, minimum 1)
        n1 = max(1, int(3 * depth_multiple))
        n2 = max(1, int(6 * depth_multiple))
        n3 = max(1, int(6 * depth_multiple))
        n4 = max(1, int(3 * depth_multiple))

        # Stem: Conv 3×3/2
        self.stem = Conv(3, c1, 3, 2)

        # Stage 1: downsample + RepC2f
        self.stage1 = nn.Sequential(
            Conv(c1, c2, 3, 2),
            RepC2f(c2, c2, num_blocks=n1, shortcut=True),
        )

        # Stage 2: downsample + C2f_CIB (more efficient for deeper stages)
        self.stage2 = nn.Sequential(
            Conv(c2, c3, 3, 2),
            C2f_CIB(c3, c3, num_blocks=n2, shortcut=True) if not use_pconv
            else self._make_pconv_stage(c3, n2),
        )

        # Stage 3: downsample + C2f_CIB
        self.stage3 = nn.Sequential(
            Conv(c3, c4, 3, 2),
            C2f_CIB(c4, c4, num_blocks=n3, shortcut=True) if not use_pconv
            else self._make_pconv_stage(c4, n3),
        )

        # Stage 4: downsample + RepC2f + SPPF
        self.stage4 = nn.Sequential(
            Conv(c4, c5, 3, 2),
            RepC2f(c5, c5, num_blocks=n4, shortcut=True),
            SPPF(c5, c5, kernel_size=5),
        )

        # Output channel counts for neck
        self.out_channels = [c3, c4, c5]  # P3, P4, P5

    def _make_pconv_stage(self, channels, num_blocks):
        """Create a stage using PConv blocks (ultra-efficient)."""
        return nn.Sequential(*[PConvBlock(channels) for _ in range(num_blocks)])

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)

        p3 = self.stage2(x)   # P3: 1/8 resolution
        p4 = self.stage3(p3)  # P4: 1/16 resolution
        p5 = self.stage4(p4)  # P5: 1/32 resolution

        return [p3, p4, p5]


# =============================================================================
# 2. BiFPN v2 — Proper Weighted Element-Wise Fusion
# =============================================================================

class BiFPNLayer(nn.Module):
    """
    Single BiFPN fusion layer with PROPER weighted element-wise fusion.

    The key difference from the existing (broken) BiFPN:
      OLD (wrong): Concat features → Conv → output (loses spatial alignment)
      NEW (correct): w₁·feat₁ + w₂·feat₂ + ... → Conv → output

    Fast normalized fusion (from EfficientDet paper):
      output = Σ(wᵢ · featᵢ) / (Σ wᵢ + ε)

    where wᵢ are LEARNABLE weights (initialized to 1) that go through
    ReLU to ensure non-negativity.

    Reference: Tan et al., "EfficientDet" (CVPR 2020) — arXiv:1911.09070
               Section 3.3, Eq. 4 (fast normalized fusion)
    """

    def __init__(self, channels, num_inputs=2, eps=1e-4):
        super().__init__()
        self.eps = eps
        # Learnable fusion weights — initialized to 1.0 (equal weighting)
        self.weights = nn.Parameter(torch.ones(num_inputs))
        self.conv = Conv(channels, channels, 3)

    def forward(self, features):
        """
        Args:
            features: list of tensors [feat₁, feat₂, ...] all same shape

        Returns:
            fused feature tensor
        """
        # Fast normalized fusion weights
        w = F.relu(self.weights)
        w_norm = w / (w.sum() + self.eps)

        # Weighted element-wise sum
        fused = sum(w_i * feat for w_i, feat in zip(w_norm, features))
        return self.conv(fused)


class BiFPNNeck(nn.Module):
    """
    BiFPN Neck — Bidirectional Feature Pyramid Network.

    Structure (2 rounds of fusion):

    Round 1 (top-down):
      P5_td = Conv(P5)
      P4_td = Fuse(P4, Upsample(P5_td))
      P3_td = Fuse(P3, Upsample(P4_td))

    Round 2 (bottom-up):
      P3_out = P3_td
      P4_out = Fuse(P4_td, P4, Downsample(P3_out))
      P5_out = Fuse(P5_td, P5, Downsample(P4_out))

    Each Fuse operation uses weighted element-wise fusion with
    learnable weights — NOT concatenation.

    Reference: Tan et al., "EfficientDet" (CVPR 2020)
    """

    def __init__(self, in_channels, out_channel=None, num_repeats=2):
        """
        Args:
            in_channels: [c3, c4, c5] — channel counts from backbone
            out_channel: output channel count (default: c3)
            num_repeats: number of BiFPN rounds
        """
        super().__init__()
        c3, c4, c5 = in_channels
        self.out_channel = out_channel or c3

        # Channel alignment (project all features to same channel count)
        self.align_p3 = Conv(c3, self.out_channel, 1) if c3 != self.out_channel else nn.Identity()
        self.align_p4 = Conv(c4, self.out_channel, 1) if c4 != self.out_channel else nn.Identity()
        self.align_p5 = Conv(c5, self.out_channel, 1) if c5 != self.out_channel else nn.Identity()

        # BiFPN rounds
        self.rounds = nn.ModuleList()
        for _ in range(num_repeats):
            self.rounds.append(nn.ModuleDict({
                # Top-down fusion
                'td_p4': BiFPNLayer(self.out_channel, num_inputs=2),
                'td_p3': BiFPNLayer(self.out_channel, num_inputs=2),
                # Bottom-up fusion
                'bu_p4': BiFPNLayer(self.out_channel, num_inputs=3),
                'bu_p5': BiFPNLayer(self.out_channel, num_inputs=3),
            }))

        self.out_channels = [self.out_channel] * 3

    def forward(self, features):
        """
        Args:
            features: [P3, P4, P5] from backbone

        Returns:
            [P3_out, P4_out, P5_out] — fused features
        """
        p3, p4, p5 = features

        # Align channels
        p3 = self.align_p3(p3)
        p4 = self.align_p4(p4)
        p5 = self.align_p5(p5)

        for round_module in self.rounds:
            # Top-down path
            p5_td = p5
            p4_up = F.interpolate(p5_td, size=p4.shape[2:], mode='nearest')
            p4_td = round_module['td_p4']([p4, p4_up])

            p3_up = F.interpolate(p4_td, size=p3.shape[2:], mode='nearest')
            p3_td = round_module['td_p3']([p3, p3_up])

            # Bottom-up path
            p3_out = p3_td
            p4_down = F.interpolate(p3_out, size=p4.shape[2:], mode='nearest')
            p4_out = round_module['bu_p4']([p4_td, p4, p4_down])

            p5_down = F.interpolate(p4_out, size=p5.shape[2:], mode='nearest')
            p5_out = round_module['bu_p5']([p5_td, p5, p5_down])

            # Update for next round
            p3, p4, p5 = p3_out, p4_out, p5_out

        return [p3, p4, p5]


# =============================================================================
# 3. Decoupled Head v2 — Improved initialization + optional shared weights
# =============================================================================

class DecoupledHeadV2(nn.Module):
    """
    Improved decoupled detection head with:
      - Better bias initialization (from YOLOX paper)
      - Optional shared stem for parameter reduction
      - DFL integration for precise box regression
    """

    def __init__(self, num_classes=80, channels=None, reg_max=16,
                 shared_stem=False):
        super().__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max
        channels = channels or [256, 256, 256]

        self.cls_branches = nn.ModuleList()
        self.reg_branches = nn.ModuleList()

        for ch in channels:
            # Classification branch
            self.cls_branches.append(nn.Sequential(
                Conv(ch, ch, 3), Conv(ch, ch, 3),
                nn.Conv2d(ch, num_classes, 1)
            ))
            # Regression branch
            self.reg_branches.append(nn.Sequential(
                Conv(ch, ch, 3), Conv(ch, ch, 3),
                nn.Conv2d(ch, 4 * reg_max, 1)
            ))

        self.dfl = DFL(reg_max) if reg_max > 1 else None
        self._init_biases()

    def _init_biases(self):
        """Initialize classification bias for stable early training.

        Prior probability p = 0.01 → bias = -log((1-p)/p) ≈ -4.6
        This prevents massive classification loss in early epochs.
        From: Lin et al., "Focal Loss" (ICCV 2017) — Section 3.3
        """
        for cls_branch in self.cls_branches:
            nn.init.constant_(cls_branch[-1].bias, -math.log((1 - 0.01) / 0.01))

    def forward(self, features):
        cls_scores = []
        bbox_preds = []

        for i, feat in enumerate(features):
            cls = self.cls_branches[i](feat)
            reg = self.reg_branches[i](feat)

            if self.dfl is not None:
                reg = self.dfl(reg)

            cls_scores.append(cls)
            bbox_preds.append(reg)

        return cls_scores, bbox_preds


import math


# =============================================================================
# 4. Badger v2 — Complete Model Assembly
# =============================================================================

class BadgerV2(nn.Module):
    """
    Badger v2: State-of-the-art object detection model.

    Combines all 2024-2026 innovations:
      - BackboneV2: PConv/RepC2f/C2f_CIB (efficient, reparameterizable)
      - BiFPNNeck: proper weighted element-wise fusion
      - DecoupledHeadV2: improved initialization
      - NMS-free option via DualHead

    Usage:
        # Standard model
        model = BadgerV2(num_classes=80, variant='small')

        # NMS-free model
        model = BadgerV2(num_classes=80, variant='small', nms_free=True)

        # Ultra-efficient edge model
        model = BadgerV2(num_classes=80, variant='nano', use_pconv=True)
    """

    def __init__(self, num_classes=80, variant='small',
                 neck_type='bifpn', nms_free=False, use_pconv=False):
        super().__init__()
        self.num_classes = num_classes
        self.variant = variant
        self.nms_free = nms_free

        # Get scaling factors
        width, depth = self._get_scale(variant)

        # Backbone
        self.backbone = BackboneV2(
            width_multiple=width,
            depth_multiple=depth,
            use_pconv=use_pconv,
        )

        # Neck
        in_channels = self.backbone.out_channels
        if neck_type == 'bifpn':
            self.neck = BiFPNNeck(in_channels, num_repeats=2)
            neck_out = self.neck.out_channels
        else:
            # Fall back to standard PA-FPN
            from .neck import PAFPN
            self.neck = PAFPN(in_channels)
            neck_out = in_channels

        # Head
        if nms_free:
            from .head import DualHead
            self.head = DualHead(num_classes=num_classes, channels=neck_out)
        else:
            self.head = DecoupledHeadV2(
                num_classes=num_classes, channels=neck_out
            )

    @staticmethod
    def _get_scale(variant):
        """Get width/depth multipliers for model variant."""
        scales = {
            'pico':   (0.125, 0.25),
            'nano':   (0.25, 0.33),
            'tiny':   (0.375, 0.33),
            'small':  (0.50, 0.33),
            'medium': (0.75, 0.67),
            'large':  (1.0, 1.0),
            'xl':     (1.25, 1.0),
        }
        if variant not in scales:
            raise ValueError(f"Unknown variant: {variant}. Options: {list(scales.keys())}")
        return scales[variant]

    def forward(self, x):
        features = self.backbone(x)
        fused = self.neck(features)
        return self.head(fused)

    def count_parameters(self):
        """Count total and trainable parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

    def fuse_for_deploy(self):
        """Fuse RepConv and Conv+BN for deployment speed."""
        fuse_model_reparam(self)
        return self


# =============================================================================
# 5. Model Factory
# =============================================================================

def create_badger_v2(variant='small', num_classes=80, neck_type='bifpn',
                     nms_free=False, use_pconv=False, pretrained=None):
    """
    Factory function for creating Badger v2 models.

    Args:
        variant: 'pico', 'nano', 'tiny', 'small', 'medium', 'large', 'xl'
        num_classes: number of detection classes
        neck_type: 'bifpn' (recommended) or 'pafpn'
        nms_free: if True, use DualHead for NMS-free inference
        use_pconv: if True, use PConv blocks (ultralight)
        pretrained: path to pretrained weights

    Returns:
        BadgerV2 model

    Examples:
        # Edge deployment (Raspberry Pi, Jetson Nano)
        model = create_badger_v2('nano', use_pconv=True)

        # Balanced accuracy/speed
        model = create_badger_v2('small', nms_free=True)

        # Maximum accuracy
        model = create_badger_v2('xl', neck_type='bifpn')
    """
    model = BadgerV2(
        num_classes=num_classes,
        variant=variant,
        neck_type=neck_type,
        nms_free=nms_free,
        use_pconv=use_pconv,
    )

    if pretrained:
        state = torch.load(pretrained, map_location='cpu')
        if 'model' in state:
            state = state['model']
        model.load_state_dict(state, strict=False)
        print(f"  Loaded pretrained weights from {pretrained}")

    return model


# =============================================================================
# 6. Quick Comparison Tool
# =============================================================================

def compare_models(variants=None, num_classes=80, device='cpu'):
    """
    Compare Badger v2 model variants side-by-side.

    Prints a table showing params, FLOPs, and latency for each variant.
    """
    
    variants = variants or ['pico', 'nano', 'tiny', 'small', 'medium', 'large']

    print(f"\n{'='*75}")
    print(f"  BADGER v2 MODEL COMPARISON")
    print(f"{'='*75}")
    print(f"  {'Variant':<10} {'Params':<12} {'GFLOPs':<10} {'Latency':<12} {'FPS':<8}")
    print(f"  {'-'*10} {'-'*12} {'-'*10} {'-'*12} {'-'*8}")

    for v in variants:
        try:
            model = create_badger_v2(v, num_classes=num_classes)
            result = profile_model(model, device=device)
            print(f"  {v:<10} {result['params_M']:<12.2f}M "
                  f"{result['flops_G']:<10.1f} "
                  f"{result['latency_ms']:<12.1f}ms "
                  f"{result['fps']:<8.0f}")
        except Exception as e:
            print(f"  {v:<10} Error: {e}")

    print(f"{'='*75}\n")
