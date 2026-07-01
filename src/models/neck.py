"""
Feature Pyramid Network (FPN) necks for multi-scale feature fusion.

The neck combines features from different backbone levels so the detector
can see both fine details (from shallow layers) and semantic meaning (from
deep layers). Badger uses a PA-FPN (Path Aggregation FPN).
"""

import torch
import torch.nn as nn
from .blocks import Conv, C2f, CSPRepLayer, make_divisible


class PAFPN(nn.Module):
    """
    Path Aggregation Feature Pyramid Network — Badger's neck.

    This is an improved FPN that adds bottom-up paths on top of the
    traditional top-down FPN. Think of it as:

      Top-down path:     P5 ──→ P4 ──→ P3   (spreads semantic info down)
      Bottom-up path:    P3 ──→ P4 ──→ P5   (spreads spatial info up)

    Architecture:
      P5 ──[upsample]──┐
                        ├─[concat]─[C2f]──→ N4 ──[upsample]──┐
      P4 ──────────────┘                                      ├─[concat]─[C2f]──→ N3
      P3 ─────────────────────────────────────────────────────┘
                                                               │
      N3 ──[conv ds]──┐                                        │
                       ├─[concat]─[C2f]──→ N4_out              │
      N4 ─────────────┘                                        │
                                                               │
      N4_out ──[conv ds]──┐                                    │
                           ├─[concat]─[C2f]──→ N5_out          │
      P5 ──────────────────┘
    """

    def __init__(self, in_channels, width_multiple=0.5, depth_multiple=0.33,
                 use_csprep=False):
        """
        Args:
            in_channels: list of input channels from backbone [c3, c4, c5]
            use_csprep: if True, use CSPRepLayer (RepVGG-based, RT-DETR style)
                       instead of C2f. Reparameterizable at inference for +speed.
        """
        super().__init__()

        self.reduce_channels = make_divisible(256 * width_multiple)
        n = lambda base: max(1, int(base * depth_multiple))
        self.use_csprep = use_csprep

        # Choose block type
        if use_csprep:
            self._fusion_block = lambda in_ch, out_ch, n_blocks: CSPRepLayer(
                in_ch, out_ch, num_blocks=n_blocks)
        else:
            self._fusion_block = lambda in_ch, out_ch, n_blocks: C2f(
                in_ch, out_ch, n_blocks, shortcut=False)

        self.c3, self.c4, self.c5 = in_channels

        # --- Top-down pathway ---
        # P5 → upsample to P4 size
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        # P5 → reduce channels → upsample → concat with P4
        self.top_down_conv1 = Conv(self.c5, self.reduce_channels, 1, 1)
        self.top_down_fuse1 = self._fusion_block(
            self.reduce_channels + self.c4, self.reduce_channels, n(3))

        # P4' → reduce channels → upsample → concat with P3
        self.top_down_conv2 = Conv(self.reduce_channels, self.reduce_channels, 1, 1)
        self.top_down_fuse2 = self._fusion_block(
            self.reduce_channels + self.c3, self.reduce_channels, n(3))

        # --- Bottom-up pathway ---
        # P3' → downsample → concat with P4'
        self.bottom_up_conv1 = Conv(self.reduce_channels, self.reduce_channels, 3, 2)
        self.bottom_up_fuse1 = self._fusion_block(
            self.reduce_channels + self.reduce_channels, self.reduce_channels, n(3))

        # P4'' → downsample → concat with P5'
        self.bottom_up_conv2 = Conv(self.reduce_channels, self.reduce_channels, 3, 2)
        self.bottom_up_fuse2 = self._fusion_block(
            self.reduce_channels + self.reduce_channels, self.reduce_channels, n(3))

        # Output channels (all same size after neck)
        self.out_channels = [self.reduce_channels, self.reduce_channels, self.reduce_channels]

    def forward(self, features):
        """
        Args:
            features: [P3, P4, P5] from backbone
                      Shapes: [B, c3, H/8, W/8], [B, c4, H/16, W/16], [B, c5, H/32, W/32]

        Returns:
            [N3_out, N4_out, N5_out] — fused features at 3 scales
        """
        p3, p4, p5 = features

        # --- Top-down ---
        # P5 pathway
        p5_reduced = self.top_down_conv1(p5)          # Reduce channels
        p5_up = self.upsample(p5_reduced)              # Upsample to P4 size
        p4_fused = torch.cat([p5_up, p4], dim=1)       # Concat
        p4_out = self.top_down_fuse1(p4_fused)          # Fuse

        # P4' pathway
        p4_reduced = self.top_down_conv2(p4_out)       # Reduce channels
        p4_up = self.upsample(p4_reduced)               # Upsample to P3 size
        p3_fused = torch.cat([p4_up, p3], dim=1)       # Concat
        n3 = self.top_down_fuse2(p3_fused)              # ← P3 output

        # --- Bottom-up ---
        # P3 → P4
        n3_down = self.bottom_up_conv1(n3)             # Downsample
        n4_fused = torch.cat([n3_down, p4_out], dim=1) # Concat
        n4 = self.bottom_up_fuse1(n4_fused)             # ← P4 output

        # P4 → P5
        n4_down = self.bottom_up_conv2(n4)             # Downsample
        n5_fused = torch.cat([n4_down, p5_reduced], dim=1)  # Concat
        n5 = self.bottom_up_fuse2(n5_fused)             # ← P5 output

        return [n3, n4, n5]


class BiFPN(nn.Module):
    """
    Bi-directional Feature Pyramid Network (EfficientDet-style).

    Improvement experiment: BiFPN adds learned weights for each feature
    fusion and removes nodes with only one input. Generally more efficient
    than PA-FPN.

    TODO: Implement fully.
    """
    def __init__(self, in_channels, width_multiple=0.5):
        super().__init__()
        self.reduce_channels = make_divisible(256 * width_multiple)
        raise NotImplementedError("BiFPN coming soon as an improvement experiment!")


def make_divisible(x, divisor=8):
    import math
    return int(math.ceil(x / divisor) * divisor)
