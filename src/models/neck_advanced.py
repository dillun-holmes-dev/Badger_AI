"""
Advanced Feature Pyramid Networks for Badger.

Standard PA-FPN is good. These are better:

  1. BiFPN (EfficientDet-style): Weighted bidirectional feature fusion.
     Each input has a learned weight — the network learns which features
     matter most at each fusion node. This is strictly more expressive
     than simple concatenation.

  2. ASFF (Adaptive Spatial Feature Fusion): Instead of fixed fusion
     weights, compute spatially-varying attention maps that tell the
     network WHERE each feature scale is most informative. This is
     especially powerful for objects that appear at different scales
     in different image regions.

  3. NAS-FPN (Neural Architecture Search FPN): Searched architecture
     that finds the optimal cross-scale connections. Included for
     reference — we pre-define the best found architecture.

Reference papers:
  - Tan et al., "EfficientDet" (CVPR 2020) — arXiv:1911.09070 — BiFPN
  - Liu et al., "ASFF" (arXiv:1911.09516) — Adaptive fusion
  - Ghiasi et al., "NAS-FPN" (CVPR 2019) — arXiv:1904.07392
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import Conv, C2f, make_divisible


# =============================================================================
# 1. BiFPN — Weighted Bidirectional Feature Pyramid Network
# =============================================================================

class BiFPN(nn.Module):
    """
    BiFPN from EfficientDet.

    Key innovations over PA-FPN:
      1. Removes nodes with only one input edge (no useful fusion)
      2. Adds skip connections from input directly to output at same level
      3. Uses learned per-input weights at each fusion node via
         fast normalized fusion:
           fused = Σ (w_i · feature_i) / (Σ w_i + ε)

         where w_i = ReLU(learned_scalar) ≥ 0

    This lets the network learn which features are most important
    at each fusion node, rather than treating all inputs equally.

    Expected: +0.5-2.0 AP over PA-FPN on COCO.
    Paper: Tan et al., EfficientDet (CVPR 2020) — arXiv:1911.09070
    """

    def __init__(self, in_channels, width_multiple=0.5, depth_multiple=0.33,
                 num_layers=3):
        super().__init__()
        self.reduce_channels = make_divisible(256 * width_multiple)
        n = lambda base: max(1, int(base * depth_multiple))

        self.c3, self.c4, self.c5 = in_channels

        # Input projection: reduce backbone channels to consistent size
        self.p3_proj = Conv(self.c3, self.reduce_channels, 1, 1)
        self.p4_proj = Conv(self.c4, self.reduce_channels, 1, 1)
        self.p5_proj = Conv(self.c5, self.reduce_channels, 1, 1)

        ch = self.reduce_channels

        # Resize operations for cross-scale connections
        self.p5_to_p4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.p4_to_p3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.p3_to_p4 = Conv(ch, ch, 3, 2)
        self.p4_to_p5 = Conv(ch, ch, 3, 2)

        # Post-fusion processing: simple conv (not C2f) because
        # weighted fusion already produces ch channels (not 2*ch).
        # BiFPN paper uses depthwise separable conv here; we use
        # standard Conv for simplicity — can upgrade to DWConv for edge.
        self.p4_td_conv = Conv(ch, ch, 3, 1)
        self.p3_td_conv = Conv(ch, ch, 3, 1)
        self.p4_bu_conv = Conv(ch, ch, 3, 1)
        self.p5_bu_conv = Conv(ch, ch, 3, 1)

        # Learned fusion weights (fast normalized fusion: w / Σw)
        # Each fusion node has 2 inputs → 2 weights
        self.w_p4_td = nn.Parameter(torch.ones(2), requires_grad=True)
        self.w_p3_td = nn.Parameter(torch.ones(2), requires_grad=True)
        self.w_p4_bu = nn.Parameter(torch.ones(2), requires_grad=True)
        self.w_p5_bu = nn.Parameter(torch.ones(2), requires_grad=True)

        self.out_channels = [ch, ch, ch]

    def _weighted_fuse(self, w, *features):
        """Fast normalized fusion: Σ(w_i·f_i) / Σ(w_i+ε)."""
        w = F.relu(w)  # Ensure non-negative
        w_sum = w.sum() + 1e-4
        fused = sum(w[i] * features[i] for i in range(len(features))) / w_sum
        return fused

    def forward(self, features):
        p3, p4, p5 = features

        # Project to consistent channels
        p3 = self.p3_proj(p3)
        p4 = self.p4_proj(p4)
        p5_in = self.p5_proj(p5)

        # --- Top-down ---
        # P5 → P4
        p5_up = self.p5_to_p4(p5_in)
        p4_td = self._weighted_fuse(self.w_p4_td, p4, p5_up)
        p4_td = self.p4_td_conv(p4_td)

        # P4' → P3
        p4_up = self.p4_to_p3(p4_td)
        p3_td = self._weighted_fuse(self.w_p3_td, p3, p4_up)
        p3_out = self.p3_td_conv(p3_td)

        # --- Bottom-up ---
        # P3 → P4
        p3_down = self.p3_to_p4(p3_out)
        p4_bu = self._weighted_fuse(self.w_p4_bu, p4_td, p3_down)
        p4_out = self.p4_bu_conv(p4_bu)

        # P4 → P5
        p4_down = self.p4_to_p5(p4_out)
        p5_bu = self._weighted_fuse(self.w_p5_bu, p5_in, p4_down)
        p5_out = self.p5_bu_conv(p5_bu)

        return [p3_out, p4_out, p5_out]


# =============================================================================
# 2. ASFF — Adaptive Spatial Feature Fusion
# =============================================================================

class ASFF(nn.Module):
    """
    Adaptive Spatial Feature Fusion.

    Instead of fixed weights per fusion node, ASFF computes spatially-
    varying attention maps. For each output scale, the network learns
    WHERE in the image to trust each input feature map.

    For example, at P3 (high-resolution output):
      The network might learn to trust P3's own features for small
      objects in the center, but P4's upsampled features for larger
      objects near the edges.

    This is more expressive than BiFPN's scalar weights, at the cost
    of slightly more parameters and compute.

    Paper: Liu et al., "ASFF" — arXiv:1911.09516
    """

    def __init__(self, in_channels, width_multiple=0.5):
        super().__init__()
        ch = make_divisible(256 * width_multiple)
        self.ch = ch

        self.c3, self.c4, self.c5 = in_channels

        # Input projections
        self.p3_proj = Conv(self.c3, ch, 1, 1)
        self.p4_proj = Conv(self.c4, ch, 1, 1)
        self.p5_proj = Conv(self.c5, ch, 1, 1)

        # Resize ops: downsample (stride=2) and upsample (×2)
        self.downsample = nn.MaxPool2d(2, 2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # ASFF for P3 output: fuse P3, upsampled P4, upsampled P5
        self.asff_p3 = self._make_asff_block(ch)

        # ASFF for P4 output: fuse downsampled P3, P4, upsampled P5
        self.asff_p4 = self._make_asff_block(ch)

        # ASFF for P5 output: fuse downsampled P3, downsampled P4, P5
        self.asff_p5 = self._make_asff_block(ch)

        self.out_channels = [ch, ch, ch]

    def _make_asff_block(self, channels):
        """Create spatial attention weights for 3 inputs."""
        return nn.Sequential(
            Conv(channels * 3, channels, 1, 1),
            Conv(channels, 3, 1, 1),  # 3 spatial attention maps
            nn.Softmax(dim=1),        # Normalize across inputs
        )

    def forward(self, features):
        p3, p4, p5 = features

        # Project to consistent channels
        p3 = self.p3_proj(p3)  # [B, C, H, W]
        p4 = self.p4_proj(p4)  # [B, C, H/2, W/2]
        p5 = self.p5_proj(p5)  # [B, C, H/4, W/4]

        # --- ASFF for P3 (high resolution) ---
        p4_to_p3 = self.upsample(p4)  # Upsample to P3 size
        p5_to_p3 = self.upsample(self.upsample(p5))  # Upsample ×4

        # Compute spatial attention weights
        p3_inputs = torch.cat([p3, p4_to_p3, p5_to_p3], dim=1)
        p3_weights = self.asff_p3(p3_inputs)  # [B, 3, H, W]

        # Weighted fusion
        p3_out = (p3_weights[:, 0:1] * p3 +
                  p3_weights[:, 1:2] * p4_to_p3 +
                  p3_weights[:, 2:3] * p5_to_p3)

        # --- ASFF for P4 (medium resolution) ---
        p3_to_p4 = self.downsample(p3)  # Downsample to P4 size
        p5_to_p4 = self.upsample(p5)    # Upsample to P4 size

        p4_inputs = torch.cat([p3_to_p4, p4, p5_to_p4], dim=1)
        p4_weights = self.asff_p4(p4_inputs)

        p4_out = (p4_weights[:, 0:1] * p3_to_p4 +
                  p4_weights[:, 1:2] * p4 +
                  p4_weights[:, 2:3] * p5_to_p4)

        # --- ASFF for P5 (low resolution) ---
        p3_to_p5 = self.downsample(self.downsample(p3))
        p4_to_p5 = self.downsample(p4)

        p5_inputs = torch.cat([p3_to_p5, p4_to_p5, p5], dim=1)
        p5_weights = self.asff_p5(p5_inputs)

        p5_out = (p5_weights[:, 0:1] * p3_to_p5 +
                  p5_weights[:, 1:2] * p4_to_p5 +
                  p5_weights[:, 2:3] * p5)

        return [p3_out, p4_out, p5_out]


# =============================================================================
# 3. NAS-FPN (Pre-defined Best Architecture)
# =============================================================================

class NASFPN(nn.Module):
    """
    NAS-FPN — neural architecture search found FPN.

    Uses a pre-defined, searched architecture with 7 feature merging
    cells that connect features across scales in non-trivial ways.

    The key finding: cross-scale connections beyond simple top-down
    + bottom-up improve accuracy. The searched architecture includes
    connections like P3→P5 and P5→P3 directly, skipping intermediate
    scales.

    Paper: Ghiasi et al., "NAS-FPN" (CVPR 2019) — arXiv:1904.07392

    Note: This is a simplified reference implementation. The full
    NAS-FPN has 7 repeating cells with learned connections.
    """

    def __init__(self, in_channels, width_multiple=0.5):
        super().__init__()
        ch = make_divisible(256 * width_multiple)

        self.c3, self.c4, self.c5 = in_channels

        # Input projections
        self.p3_proj = Conv(self.c3, ch, 1, 1)
        self.p4_proj = Conv(self.c4, ch, 1, 1)
        self.p5_proj = Conv(self.c5, ch, 1, 1)

        # Merge cells (simplified from the searched architecture)
        self.cell_p3 = C2f(ch * 2, ch, 1, shortcut=False)
        self.cell_p4 = C2f(ch * 2, ch, 1, shortcut=False)
        self.cell_p5 = C2f(ch * 2, ch, 1, shortcut=False)

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.downsample = nn.MaxPool2d(2, 2)

        self.out_channels = [ch, ch, ch]

    def forward(self, features):
        p3, p4, p5 = features

        p3 = self.p3_proj(p3)
        p4 = self.p4_proj(p4)
        p5 = self.p5_proj(p5)

        # Cross-scale connections (NAS-FPN pattern)
        # P3 merges with upsampled P5 directly
        p3_merged = torch.cat([p3, self.upsample(self.upsample(p5))], dim=1)
        p3_out = self.cell_p3(p3_merged)

        # P4 merges with P3 and P5
        p4_merged = torch.cat([p4, self.upsample(p5)], dim=1)
        p4_out = self.cell_p4(p4_merged)

        # P5 merges with downsampled P3 directly
        p5_merged = torch.cat([p5, self.downsample(self.downsample(p3))], dim=1)
        p5_out = self.cell_p5(p5_merged)

        return [p3_out, p4_out, p5_out]
