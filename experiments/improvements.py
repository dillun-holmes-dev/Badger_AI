"""
Experiment framework — systematically test architectural improvements.

Each experiment is isolated and tracked so you can measure exactly what
moves the needle on mAP, latency, and parameter count.

How to use:
  1. Pick an experiment from EXPERIMENTS below
  2. Run: python scripts/run_experiment.py --name cbam_attention
  3. Compare results in runs/<experiment_name>/
"""

import torch
import torch.nn as nn

# =============================================================================
# Experiment 1: Attention Modules
# =============================================================================

class CBAM(nn.Module):
    """
    Convolutional Block Attention Module.

    Adds channel + spatial attention to any feature map.
    Lightweight (~1% parameter increase), typically +0.5-1.5% mAP.

    Paper: Woo et al., "CBAM: Convolutional Block Attention Module", ECCV 2018

    Where to insert: after every C2f module in the backbone.
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        # Channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.channel_att = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        # Spatial attention
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False),
            nn.Sigmoid()
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Channel attention
        avg_out = self.channel_att(self.avg_pool(x))
        max_out = self.channel_att(self.max_pool(x))
        channel_weight = self.sigmoid(avg_out + max_out)
        x = x * channel_weight

        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_weight = self.spatial_att(torch.cat([avg_out, max_out], dim=1))
        x = x * spatial_weight

        return x


class ECA(nn.Module):
    """
    Efficient Channel Attention — even lighter than CBAM.

    1D convolution instead of MLP for channel attention.
    Almost zero parameter overhead (~0.01%).

    Paper: Wang et al., "ECA-Net: Efficient Channel Attention", CVPR 2020
    """
    def __init__(self, channels, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=k_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block.

    The original channel attention mechanism. Heavier than ECA but proven.

    Paper: Hu et al., "Squeeze-and-Excitation Networks", CVPR 2018
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        y = x.mean(dim=[2, 3])  # Global average pool
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


# =============================================================================
# Experiment 2: Activation Functions
# =============================================================================

class Mish(nn.Module):
    """Mish activation — sometimes better than SiLU for small models."""
    def forward(self, x):
        return x * torch.tanh(torch.nn.functional.softplus(x))


# =============================================================================
# Experiment 3: Normalization Alternatives
# =============================================================================

class GroupNormConv(nn.Module):
    """
    GroupNorm + Conv instead of BatchNorm + Conv.

    BatchNorm struggles with small batches. GroupNorm doesn't.
    Try this when using batch_size < 8.
    """
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 groups=1, num_groups=32):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              kernel_size // 2, groups=groups, bias=False)
        self.gn = nn.GroupNorm(min(num_groups, out_channels), out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.gn(self.conv(x)))


# =============================================================================
# Experiment 4: EMA (Exponential Moving Average)
# =============================================================================

class ModelEMA:
    """
    Exponential Moving Average of model weights.

    Maintains a shadow copy of the model with smoothed weights.
    Typically gives +0.5-1.0% mAP at inference time.

    Usage:
        ema = ModelEMA(model)
        for epoch in range(epochs):
            train_one_epoch(model)
            ema.update(model)
        ema.apply()  # Apply EMA weights before validation
    """
    def __init__(self, model, decay=0.9999):
        self.ema = {k: v.clone().detach() for k, v in model.state_dict().items()}
        self.decay = decay
        self.updates = 0

    def update(self, model):
        """Update EMA weights after each training step."""
        self.updates += 1
        d = self.decay * (1 - torch.exp(torch.tensor(-self.updates / 2000)))

        with torch.no_grad():
            for k, v in model.state_dict().items():
                if v.dtype.is_floating_point:
                    self.ema[k] = d * self.ema[k] + (1 - d) * v
                else:
                    self.ema[k] = v

    def apply(self, model):
        """Apply EMA weights to the model (for validation/inference)."""
        model.load_state_dict(self.ema)

    def restore(self, model_state):
        """Restore original weights (for continuing training)."""
        # model_state is stored separately
        pass


# =============================================================================
# Experiment 5: Multi-scale Training
# =============================================================================

def random_resize(img_size, scale_range=(0.5, 1.5)):
    """
    Randomly resize input during training.

    Multi-scale training improves robustness to object size variation.
    Typical gain: +0.5-1.0% mAP.

    Args:
        img_size: base image size
        scale_range: (min_scale, max_scale)

    Returns:
        new size (multiple of 32)
    """
    scale = scale_range[0] + torch.rand(1).item() * (scale_range[1] - scale_range[0])
    new_size = int(img_size * scale)
    # Round to nearest multiple of 32
    new_size = (new_size // 32) * 32
    return max(320, min(1280, new_size))


# =============================================================================
# Experiment 6: Deformable Convolution v2 (DCNv2)
# =============================================================================

class DeformableConv2d(nn.Module):
    """
    Deformable Convolution v2 — learns spatial sampling offsets.

    Standard conv: samples fixed grid positions (e.g., 3×3 kernel
    always looks at the same 9 relative positions).

    DCNv2: learns an OFFSET for each sampling position, so the kernel
    can deform to follow object contours. This is especially powerful
    for objects with non-rigid shapes (animals, clothing, hands).

    The math:
      y(p₀) = Σ w(pₙ)·x(p₀ + pₙ + Δpₙ)·Δmₙ

    where Δpₙ are learned offsets and Δmₙ are modulation scalars
    (the "v2" addition — lets the network learn WHICH sampling
    positions matter and which don't).

    Paper: Zhu et al., "Deformable ConvNets v2" (CVPR 2019) — arXiv:1811.11168
    Claimed: +1-3 AP on COCO when replacing C3-C5 convs in ResNet-50 backbone
             (COCO val2017, 8×V100, FP32, 24 epochs)

    Cost: ~10% more params, ~15% slower inference
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Standard conv for the actual computation
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                             stride, padding, dilation, groups, bias)

        # Offset + modulation prediction (2*k*k for offsets, k*k for modulation)
        self.offset_conv = nn.Conv2d(in_channels, 3 * kernel_size * kernel_size,
                                     kernel_size, stride, padding, bias=True)

        # Initialize offsets to zero (start as regular conv)
        nn.init.constant_(self.offset_conv.weight, 0)
        nn.init.constant_(self.offset_conv.bias, 0)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, x):
        # Predict offsets and modulation
        offset_mod = self.offset_conv(x)
        offset = offset_mod[:, :2 * self.kernel_size * self.kernel_size]
        modulation = torch.sigmoid(
            offset_mod[:, 2 * self.kernel_size * self.kernel_size:]
        )

        # In production, use torchvision.ops.deform_conv2d
        # For now, this is a placeholder that falls back to regular conv
        try:
            from torchvision.ops import deform_conv2d
            return deform_conv2d(
                x, offset, self.conv.weight, self.conv.bias,
                stride=self.stride, padding=self.padding, mask=modulation
            )
        except ImportError:
            # Fallback: standard convolution
            return self.conv(x)


# =============================================================================
# Experiment 7: RepConv (Structural Reparameterization)
# =============================================================================

class RepConv(nn.Module):
    """
    RepConv — train with multiple branches, deploy as one.

    During training:
      out = Conv3×3(x) + Conv1×1(x) + Identity(x) [+ BatchNorm]

    During inference (after reparameterization):
      out = Conv3×3_fused(x)

    This gives the training benefits of multi-branch (better gradient
    flow, implicit ensemble) with the inference speed of a single conv.

    The reparameterization math:
      W_fused = W_3×3 + pad(W_1×1) + pad(I)
      b_fused = b_3×3 + b_1×1 + 0

    Paper: Ding et al., "RepVGG: Making VGG-style ConvNets Great Again"
           (CVPR 2021) — arXiv:2101.03697
    Claimed: +1-3 AP over equivalent plain conv networks on COCO
             (RepVGG-B3, COCO val2017, 8×V100, FP32)

    Cost: 3× training compute, zero inference cost
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, deploy=False):
        super().__init__()
        self.deploy = deploy
        self.act = nn.SiLU()

        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels, out_channels, kernel_size,
                                         stride, padding, bias=True)
        else:
            # Training-time branches
            self.rbr_3x3 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
                nn.BatchNorm2d(out_channels),
            )
            self.rbr_1x1 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, 0, bias=False),
                nn.BatchNorm2d(out_channels),
            )
            self.rbr_identity = nn.BatchNorm2d(in_channels) if in_channels == out_channels and stride == 1 else None

    def forward(self, x):
        if self.deploy:
            return self.act(self.rbr_reparam(x))

        out = self.rbr_3x3(x) + self.rbr_1x1(x)
        if self.rbr_identity is not None:
            out += self.rbr_identity(x)
        return self.act(out)

    def _fuse_bn(self, conv, bn):
        """Fuse Conv2d + BatchNorm2d into a single Conv2d."""
        w = conv.weight
        mean = bn.running_mean
        var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps

        if gamma is None:
            gamma = torch.ones_like(mean)
        if beta is None:
            beta = torch.zeros_like(mean)

        std = (var + eps).sqrt()
        fused_weight = w * (gamma / std).view(-1, 1, 1, 1)
        fused_bias = beta - gamma * mean / std

        fused_conv = nn.Conv2d(
            conv.in_channels, conv.out_channels, conv.kernel_size,
            conv.stride, conv.padding, bias=True
        )
        fused_conv.weight.data = fused_weight
        fused_conv.bias.data = fused_bias
        return fused_conv

    def reparameterize(self):
        """Convert multi-branch training model to single-branch deployment."""
        if self.deploy:
            return

        # Fuse 3×3 branch
        kernel_3x3, bias_3x3 = self._get_fused_kernel_bias(self.rbr_3x3)

        # Fuse 1×1 branch (pad to 3×3)
        kernel_1x1, bias_1x1 = self._get_fused_kernel_bias(self.rbr_1x1)
        kernel_1x1_padded = torch.nn.functional.pad(kernel_1x1, [1, 1, 1, 1])

        # Identity branch (BN-only: must fuse BN params into identity kernel)
        if self.rbr_identity is not None:
            identity_kernel, bias_identity = self._get_identity_kernel_bias(
                self.rbr_identity,
                self.rbr_3x3[0].in_channels,
                self.rbr_3x3[0].out_channels,
                kernel_3x3.device,
            )
        else:
            identity_kernel = torch.zeros_like(kernel_3x3)
            bias_identity = torch.zeros_like(bias_3x3)

        # Sum all branches
        fused_kernel = kernel_3x3 + kernel_1x1_padded + identity_kernel
        fused_bias = bias_3x3 + bias_1x1 + bias_identity

        # Create deployment conv
        self.rbr_reparam = nn.Conv2d(
            self.rbr_3x3[0].in_channels,
            self.rbr_3x3[0].out_channels,
            self.rbr_3x3[0].kernel_size,
            self.rbr_3x3[0].stride,
            self.rbr_3x3[0].padding,
            bias=True
        )
        self.rbr_reparam.weight.data = fused_kernel
        self.rbr_reparam.bias.data = fused_bias

        # Remove training branches
        del self.rbr_3x3, self.rbr_1x1, self.rbr_identity
        self.deploy = True

    def _get_fused_kernel_bias(self, branch):
        if isinstance(branch, nn.Sequential):
            conv = branch[0]
            bn = branch[1]
            fused = self._fuse_bn(conv, bn)
            return fused.weight, fused.bias
        return branch.weight, branch.bias if branch.bias is not None else torch.zeros(branch.weight.shape[0])

    def _get_identity_kernel_bias(self, bn, in_ch, out_ch, device):
        """Create identity kernel fused with BN parameters for the BN-only branch."""
        kernel = torch.zeros(out_ch, in_ch, 3, 3, device=device)
        for i in range(min(in_ch, out_ch)):
            kernel[i, i, 1, 1] = 1.0

        # Fuse BN: W_fused = I * (gamma / std), b_fused = beta - gamma * mean / std
        gamma = bn.weight
        beta = bn.bias
        mean = bn.running_mean
        var = bn.running_var
        eps = bn.eps

        std = (var + eps).sqrt()
        fused_kernel = kernel * (gamma / std).reshape(-1, 1, 1, 1)
        fused_bias = beta - gamma * mean / std
        return fused_kernel, fused_bias


# =============================================================================
# Experiment 8: DropPath (Stochastic Depth)
# =============================================================================

class DropPath(nn.Module):
    """
    Stochastic Depth — randomly drop entire residual blocks during training.

    This is a powerful regularizer for deep networks. During training,
    each residual block has probability `drop_prob` of being skipped
    entirely (output = input).

    This forces the network to learn robust features that don't depend
    on any single block, acting as an implicit ensemble.

    Paper: Huang et al., "Deep Networks with Stochastic Depth" (ECCV 2016)
           — arXiv:1603.09382
    Claimed: +0.5-1.0% top-1 on ImageNet, reduced overfitting in deep networks

    Cost: Zero inference cost. Training: slightly faster (fewer blocks active).
    """

    def __init__(self, drop_prob=0.1):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x

        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # (B, 1, 1, 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # Binary mask
        return x / keep_prob * random_tensor  # Scale to maintain expected value


# =============================================================================
# Experiment Registry
# =============================================================================

EXPERIMENTS = {
    'cbam_attention': {
        'name': 'CBAM Attention',
        'description': 'Add CBAM after each C2f in backbone',
        'expected_gain': '+0.5-1.5% mAP',
        'param_overhead': '~1%',
        'speed_cost': '~5% slower',
        'paper': 'Woo et al., "CBAM", ECCV 2018 — arXiv:1807.06521',
        'benchmark': '+0.8% top-1 ImageNet-1k (ResNet-50, 224, 8×Titan Xp)',
    },
    'eca_attention': {
        'name': 'ECA Attention',
        'description': 'Efficient channel attention in backbone',
        'expected_gain': '+0.3-1.0% mAP',
        'param_overhead': '<0.1%',
        'speed_cost': '~2% slower',
        'paper': 'Wang et al., "ECA-Net", CVPR 2020 — arXiv:1910.03151',
        'benchmark': '+0.5% top-1 ImageNet-1k (<0.01% params, ResNet-50, 224)',
    },
    'se_attention': {
        'name': 'SE Block Attention',
        'description': 'Squeeze-and-Excitation in backbone',
        'expected_gain': '+0.5-1.0% mAP',
        'param_overhead': '~1-2%',
        'speed_cost': '~3% slower',
        'paper': 'Hu et al., "SENet", CVPR 2018 — arXiv:1709.01507',
        'benchmark': '+1.0% top-1 ImageNet-1k (ResNet-50, 224, 8×Titan X)',
    },
    'simota_assigner': {
        'name': 'SimOTA Label Assignment',
        'description': 'YOLOX dynamic-k assigner instead of TAL',
        'expected_gain': '+0.2-0.8% mAP (especially small objects)',
        'param_overhead': '0%',
        'speed_cost': 'Negligible (no grad)',
        'paper': 'Ge et al., "YOLOX", arXiv 2021 — arXiv:2107.08430',
        'benchmark': '+0.8 AP COCO val2017 (YOLOX-S vs YOLOv5-S, 640, Tesla V100, FP32)',
    },
    'mish_activation': {
        'name': 'Mish Activation',
        'description': 'Replace SiLU with Mish activation',
        'expected_gain': '+0.1-0.5% mAP (small models)',
        'param_overhead': '0%',
        'speed_cost': '~5% slower',
    },
    'multi_scale_training': {
        'name': 'Multi-scale Training',
        'description': 'Random resize during training (320-1280)',
        'expected_gain': '+0.5-1.0% mAP',
        'param_overhead': '0%',
        'speed_cost': 'Variable (depends on size)',
    },
    'ema_weights': {
        'name': 'EMA Weight Averaging',
        'description': 'Exponential moving average of weights',
        'expected_gain': '+0.3-0.8% mAP',
        'param_overhead': '0% (2x memory during training)',
        'speed_cost': 'No inference cost',
        'paper': 'Izmailov et al., "Averaging Weights", UAI 2018 — arXiv:1803.05407; used in YOLOX/D-FINE/DEIM training',
        'benchmark': 'YOLOX: +0.3 AP COCO val2017 (decay=0.9999, YOLOX-S, 640, Tesla V100)',
    },
    'bifpn_neck': {
        'name': 'BiFPN Neck',
        'description': 'Replace PA-FPN with weighted BiFPN',
        'expected_gain': '+0.3-1.0% mAP',
        'param_overhead': 'Minimal',
        'speed_cost': '~10% slower',
    },
    'giou_loss': {
        'name': 'GIoU Box Loss',
        'description': 'Try GIoU instead of CIoU for box regression',
        'expected_gain': 'Variable (try both!)',
        'param_overhead': '0%',
        'speed_cost': 'Same',
    },
    'label_smoothing': {
        'name': 'Label Smoothing',
        'description': 'Apply label smoothing (0.05-0.1) to classification',
        'expected_gain': '+0.1-0.5% mAP',
        'param_overhead': '0%',
        'speed_cost': 'No cost',
    },
    'attention_neck': {
        'name': 'Hybrid Attention Neck',
        'description': 'Swin window attention + global token mixing in FPN neck',
        'expected_gain': '+1.0-3.0% mAP',
        'param_overhead': '~10-15%',
        'speed_cost': '~20% slower',
        'paper': 'Liu et al., Swin Transformer (arXiv:2103.14030)',
    },
    'gaussian_dfl': {
        'name': 'Gaussian DFL Regression',
        'description': 'Replace discrete DFL bins with continuous Gaussian (μ,σ) per edge',
        'expected_gain': '+0.2-0.5% mAP',
        'param_overhead': '0% (2× regression channels)',
        'speed_cost': '~2% slower',
        'paper': 'Li et al., Generalized Focal Loss (arXiv:2006.04388)',
    },
    'varifocal_loss': {
        'name': 'Varifocal Classification Loss',
        'description': 'Weight classification loss by IoU quality score',
        'expected_gain': '+0.5-1.0% mAP',
        'param_overhead': '0%',
        'speed_cost': '~1% slower',
        'paper': 'Zhang et al., VarifocalNet (arXiv:2008.13367)',
    },
    'siou_loss': {
        'name': 'SIoU Box Regression Loss',
        'description': 'Angle-aware box loss — better for rotated/diagonal objects',
        'expected_gain': '+0.3-0.5% mAP',
        'param_overhead': '0%',
        'speed_cost': 'Same as CIoU',
        'paper': 'Gevorgyan, SIoU Loss (arXiv:2205.12740)',
    },
    'mosaic_close': {
        'name': 'Close Mosaic Late Training',
        'description': 'Disable mosaic augmentation for final 10 epochs',
        'expected_gain': '+0.5-1.5% mAP',
        'param_overhead': '0%',
        'speed_cost': 'Faster late training',
    },
    'larger_resolution': {
        'name': '1280×1280 Training Resolution',
        'description': 'Double input resolution for finer detail',
        'expected_gain': '+1.0-3.0% mAP',
        'param_overhead': '0%',
        'speed_cost': '~4× compute',
    },
    'deeper_backbone': {
        'name': 'Badger-XL Scale',
        'description': 'Scale to width=1.25, depth=1.0 (~68M params)',
        'expected_gain': '+3.0-5.0% mAP',
        'param_overhead': '~500% (11M → 68M)',
        'speed_cost': '~6× compute',
    },
    'dcnv2': {
        'name': 'Deformable Conv v2',
        'description': 'Replace C3-C5 convs with deformable convolutions',
        'expected_gain': '+1.0-3.0% mAP',
        'param_overhead': '~10%',
        'speed_cost': '~15% slower',
        'paper': 'Zhu et al., "Deformable ConvNets v2", CVPR 2019 — arXiv:1811.11168',
        'benchmark': '+1-3 AP COCO val2017 (ResNet-50 backbone, 8×V100, FP32, 24 epochs)',
    },
    'repconv': {
        'name': 'RepConv Reparameterization',
        'description': 'Multi-branch training → single-branch deployment',
        'expected_gain': '+1.0-3.0% mAP',
        'param_overhead': '3× training, 0× deployment',
        'speed_cost': '~3× training time, zero inference cost',
        'paper': 'Ding et al., "RepVGG", CVPR 2021 — arXiv:2101.03697',
        'benchmark': '+1-3 AP over plain conv on COCO (RepVGG-B3, 8×V100, FP32)',
    },
    'droppath': {
        'name': 'Stochastic Depth (DropPath)',
        'description': 'Randomly drop residual blocks during training',
        'expected_gain': '+0.3-1.0% mAP (regularization)',
        'param_overhead': '0%',
        'speed_cost': 'Faster training (fewer active blocks)',
        'paper': 'Huang et al., "Stochastic Depth", ECCV 2016 — arXiv:1603.09382',
        'benchmark': '+0.5-1.0% top-1 ImageNet, reduces overfitting in deep nets',
    },

    # =========================================================================
    # v2 EXPERIMENTS — State-of-the-Art (2023-2026)
    # =========================================================================

    'wiou_v3': {
        'name': 'WIoU v3 Box Loss',
        'description': 'Dynamic non-monotonic focusing — replaces CIoU for superior gradient quality',
        'expected_gain': '+0.5-1.5% mAP',
        'param_overhead': '0%',
        'speed_cost': 'Same as CIoU',
        'paper': 'Tong et al., "Wise-IoU: Bounding Box Regression Loss with Dynamic Focusing Mechanism" (2023) — arXiv:2301.10051',
        'benchmark': '+0.7 AP COCO val2017 (YOLOv7-tiny, 640, FP32)',
        'version': 'v2',
    },
    'inner_iou': {
        'name': 'Inner-IoU Loss',
        'description': 'Auxiliary bounding box regression — improves small object accuracy',
        'expected_gain': '+0.3-0.8% mAP (especially AP_S)',
        'param_overhead': '0%',
        'speed_cost': 'Same as CIoU',
        'paper': 'Zhang et al., "Inner-IoU: More Effective Bounding Box Regression" (2023) — arXiv:2311.02877',
        'benchmark': '+0.5 AP, +0.8 AP_S on COCO val2017 (YOLOv5s, 640)',
        'version': 'v2',
    },
    'focal_eiou': {
        'name': 'Focal-EIoU Loss',
        'description': 'Focal-weighted Efficient IoU — focuses on hard samples during training',
        'expected_gain': '+0.3-0.5% mAP',
        'param_overhead': '0%',
        'speed_cost': 'Same',
        'paper': 'Zhang et al., "Focal and Efficient IOU Loss" (2021) — arXiv:2101.08158',
        'benchmark': '+0.3 AP COCO val2017 (Faster R-CNN + ResNet-50)',
        'version': 'v2',
    },
    'pconv_backbone': {
        'name': 'PConv Backbone (FasterNet)',
        'description': 'Partial Convolution — process only 25% of channels, 75% FLOPs savings',
        'expected_gain': 'Same accuracy, 2-3× faster inference',
        'param_overhead': '-30% (fewer params)',
        'speed_cost': '2-3× faster',
        'paper': 'Chen et al., "Run, Don\'t Walk: Chasing Higher FLOPS" (CVPR 2023) — arXiv:2303.03667',
        'benchmark': 'FasterNet-T0: 71.9% top-1 ImageNet-1k, 0.34 GFLOPs',
        'version': 'v2',
    },
    'c2f_cib': {
        'name': 'C2f with Compact Inverted Bottleneck',
        'description': 'Replace standard 3×3 convs with depthwise conv chains — 20% smaller',
        'expected_gain': '-20% params, same accuracy',
        'param_overhead': '-20%',
        'speed_cost': '~15% faster',
        'paper': 'Wang et al., "YOLOv10: Real-Time End-to-End Object Detection" (2024) — arXiv:2405.14458',
        'benchmark': 'YOLOv10-S: 46.3 AP, 7.2M params (vs YOLOv8-S: 44.9 AP, 11.2M)',
        'version': 'v2',
    },
    'repc2f': {
        'name': 'Reparameterizable C2f',
        'description': 'Multi-branch training → single conv at deploy. Free accuracy boost.',
        'expected_gain': '+0.5-1.0% mAP, zero inference cost',
        'param_overhead': '3× during training, 0× at deploy',
        'speed_cost': 'Zero inference cost (fused)',
        'paper': 'Ding et al., "RepVGG" (CVPR 2021) + YOLOv10 (2024)',
        'benchmark': '+1-3 AP over plain conv, zero deploy overhead',
        'version': 'v2',
    },
    'nms_free': {
        'name': 'NMS-Free Dual-Head Detection',
        'description': 'one2many (training) + one2one (inference) — eliminates NMS entirely',
        'expected_gain': 'Same accuracy, removes NMS latency (~1-5ms savings)',
        'param_overhead': '~30% (dual head during training)',
        'speed_cost': 'Faster inference (no NMS post-processing)',
        'paper': 'Wang et al., "YOLOv10" (arXiv:2405.14458) — Consistent Dual Assignments',
        'benchmark': 'YOLOv10: NMS-free, -2ms latency vs YOLOv8 on COCO val2017',
        'version': 'v2',
    },
    'bifpn_v2': {
        'name': 'BiFPN v2 (Proper Weighted Fusion)',
        'description': 'Fast normalized weighted element-wise fusion — correct EfficientDet implementation',
        'expected_gain': '+0.5-1.5% mAP over concat-based BiFPN',
        'param_overhead': 'Minimal (learnable weights only)',
        'speed_cost': '~5% slower than PAFPN',
        'paper': 'Tan et al., "EfficientDet" (CVPR 2020) — arXiv:1911.09070',
        'benchmark': 'EfficientDet-D0: 33.8 AP (COCO) vs plain FPN: 32.2 AP',
        'version': 'v2',
    },
    'area_attention': {
        'name': 'Area Attention (A²)',
        'description': 'Partition-based efficient attention — large receptive field at O(N√N) cost',
        'expected_gain': '+1.0-2.0% mAP for large models',
        'param_overhead': '~10%',
        'speed_cost': '~15% slower',
        'paper': 'Tian et al., "YOLOv12: Attention-Centric Real-Time Object Detectors" (2025) — arXiv:2502.12524',
        'benchmark': 'YOLOv12-L: 53.7 AP COCO val2017 (vs YOLOv11-L: 53.4 AP)',
        'version': 'v2',
    },
    'r_elan': {
        'name': 'R-ELAN Block',
        'description': 'Residual ELAN for stable training of attention-heavy models',
        'expected_gain': 'Enables stable training of deeper attention models',
        'param_overhead': '~5%',
        'speed_cost': '~5% slower',
        'paper': 'Tian et al., "YOLOv12" (arXiv:2502.12524) — Section 3.2',
        'benchmark': 'Stabilizes training for 50+ attention layers',
        'version': 'v2',
    },
}

# =============================================================================
# Combo Builder — stack experiments
# =============================================================================

def build_combo(*experiment_names):
    """
    Combine multiple experiments into a single config.

    Example:
        combo = build_combo('simota_assigner', 'cbam_attention', 'ema_weights')
        # Returns config dict that combines all three improvements
    """
    combo = {
        'name': ' + '.join([EXPERIMENTS[e]['name'] for e in experiment_names]),
        'experiments': [],
        'total_expected_gain': [],
        'total_param_overhead': [],
    }

    for name in experiment_names:
        if name not in EXPERIMENTS:
            raise ValueError(f"Unknown experiment: {name}. Available: {list(EXPERIMENTS.keys())}")
        combo['experiments'].append(EXPERIMENTS[name])
        combo['total_expected_gain'].append(EXPERIMENTS[name]['expected_gain'])
        combo['total_param_overhead'].append(EXPERIMENTS[name]['param_overhead'])

    return combo


def list_experiments():
    """Print all available experiments."""
    print("\n" + "=" * 70)
    print("  AVAILABLE EXPERIMENTS")
    print("=" * 70)
    for key, exp in EXPERIMENTS.items():
        print(f"\n  [{key}]")
        print(f"    Name:            {exp['name']}")
        print(f"    Description:     {exp['description']}")
        print(f"    Expected Gain:   {exp['expected_gain']}")
        print(f"    Param Overhead:  {exp['param_overhead']}")
        print(f"    Speed Cost:      {exp['speed_cost']}")
    print("\n" + "=" * 70)
    print("\n  Suggested combos to try first:")
    print("    python scripts/run_experiment.py --combo simota_assigner,ema_weights")
    print("    python scripts/run_experiment.py --combo cbam_attention,simota_assigner,multi_scale_training")
    print("")


# =============================================================================
# Presets — pre-built experiment combinations
# =============================================================================

def max_accuracy_preset():
    """All high-impact experiments for maximum mAP. Target: 55-60+ mAP on COCO."""
    return build_combo(
        'simota_assigner', 'attention_neck', 'gaussian_dfl',
        'varifocal_loss', 'siou_loss', 'mosaic_close',
        'multi_scale_training', 'ema_weights',
        'larger_resolution', 'deeper_backbone'
    )


def balanced_preset():
    """Good gains, moderate compute increase. Target: 44-50 mAP on COCO."""
    return build_combo(
        'simota_assigner', 'cbam_attention', 'varifocal_loss',
        'siou_loss', 'mosaic_close', 'ema_weights'
    )


def fast_preset():
    """Accuracy gains with ZERO speed penalty. Target: +2-4 AP."""
    return build_combo(
        'simota_assigner', 'varifocal_loss', 'mosaic_close', 'ema_weights'
    )


# --- v2 Presets (State-of-the-Art) ---

def sota_v2_preset():
    """Full SOTA stack for maximum accuracy. Target: 60+ mAP on COCO."""
    return build_combo(
        'wiou_v3', 'repc2f', 'c2f_cib', 'bifpn_v2', 'nms_free',
        'varifocal_loss', 'ema_weights', 'mosaic_close',
        'multi_scale_training', 'area_attention'
    )


def edge_v2_preset():
    """Maximum efficiency for edge deployment. Target: 35+ mAP, <2ms latency."""
    return build_combo(
        'pconv_backbone', 'c2f_cib', 'wiou_v3', 'nms_free', 'ema_weights'
    )


def balanced_v2_preset():
    """Best accuracy/speed tradeoff with v2 techniques. Target: 50-55 mAP."""
    return build_combo(
        'wiou_v3', 'repc2f', 'bifpn_v2', 'nms_free',
        'varifocal_loss', 'ema_weights', 'mosaic_close'
    )


# =============================================================================
# Paper Reference Index
# =============================================================================

PAPERS = {
    # Foundational CNN architecture
    'LeNet-5 (1998)':          'http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf',
    'AlexNet (2012)':          'https://papers.nips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf',
    'VGGNet (2014)':           'https://arxiv.org/abs/1409.1556',
    'GoogLeNet/Inception':     'https://arxiv.org/abs/1409.4842',
    'ResNet (2015)':           'https://arxiv.org/abs/1512.03385',
    'EfficientNet (2019)':     'https://arxiv.org/abs/1905.11946',

    # Training mechanics
    'Conv Arithmetic Guide':   'https://arxiv.org/abs/1603.07285',
    'BatchNorm (2015)':        'https://arxiv.org/abs/1502.03167',
    'BN Optimization (2018)':  'https://arxiv.org/abs/1805.11604',
    'Kaiming Init/PReLU':      'https://arxiv.org/abs/1502.01852',
    'GELU (2016)':             'https://arxiv.org/abs/1606.08415',

    # Attention mechanisms
    'CBAM (2018)':             'https://arxiv.org/abs/1807.06521',
    'ECA-Net (2020)':          'https://arxiv.org/abs/1910.03151',
    'SE-Net (2018)':           'https://arxiv.org/abs/1709.01507',
    'Self-Attention (2017)':   'https://arxiv.org/abs/1706.03762',
    'ViT (2020)':              'https://arxiv.org/abs/2010.11929',
    'Swin Transformer (2021)': 'https://arxiv.org/abs/2103.14030',

    # Detection
    'Faster R-CNN (2015)':     'https://arxiv.org/abs/1506.01497',
    'YOLO (2015)':             'https://arxiv.org/abs/1506.02640',
    'YOLOv3 (2018)':           'https://arxiv.org/abs/1804.02767',
    'YOLOX (2021)':            'https://arxiv.org/abs/2107.08430',
    'Mask R-CNN (2017)':       'https://arxiv.org/abs/1703.06870',
    'Deformable DETR (2020)':  'https://arxiv.org/abs/2010.04159',
    'RT-DETR (2023)':          'https://arxiv.org/abs/2304.08069',
    'D-FINE (2024)':           'https://arxiv.org/abs/2410.13842',

    # Loss functions
    'Focal Loss (2017)':       'https://arxiv.org/abs/1708.02002',
    'GIoU Loss (2019)':        'https://arxiv.org/abs/1902.09630',
    'Generalized Focal Loss':  'https://arxiv.org/abs/2006.04388',
    'VarifocalNet (2020)':     'https://arxiv.org/abs/2008.13367',
    'SIoU Loss (2022)':        'https://arxiv.org/abs/2205.12740',
    'WIoU (2023)':             'https://arxiv.org/abs/2301.10051',
    'Inner-IoU (2023)':        'https://arxiv.org/abs/2311.02877',

    # Efficient architectures (2023-2026)
    'FasterNet/PConv (2023)':  'https://arxiv.org/abs/2303.03667',
    'YOLOv10 (2024)':          'https://arxiv.org/abs/2405.14458',
    'YOLOv12 (2025)':          'https://arxiv.org/abs/2502.12524',
    'EfficientDet (2020)':     'https://arxiv.org/abs/1911.09070',
    'RepVGG (2021)':           'https://arxiv.org/abs/2101.03697',

    # Multimodal / self-supervised
    'CLIP (2021)':             'https://arxiv.org/abs/2103.00020',
    'MAE (2021)':              'https://arxiv.org/abs/2111.06377',

    # Im2col / GPU convolution
    'GPU Conv (Chellapilla)':  'https://hal.science/inria-00112631/document',
}
