"""
Backbone networks for feature extraction.

The backbone is the "eyes" of Badger — it processes the raw image and extracts
meaningful features at multiple scales. Badger uses CSPDarknet, a variant of
Darknet-53 with cross-stage partial connections.
"""

import torch
import torch.nn as nn
from .blocks import Conv, C2f, SPPF, make_divisible


class CSPDarknet(nn.Module):
    """
    CSPDarknet — Badger's backbone.

    Architecture (for 640×640 input):
      Layer        Output Size    Channels
      ─────────────────────────────────────
      Conv(3→c1)   640×640        c1 (e.g., 64)
      Conv(c1→c2)  320×320        c2 (e.g., 128)
      C2f(c2→c3)   160×160        c3 (e.g., 256)
      Conv(c3→c4)  80×80          c4 (e.g., 256)  ← P3 output
      C2f(c4→c4') 80×80           c4'
      Conv(c4'→c5) 40×40          c5 (e.g., 512)  ← P4 output
      C2f(c5→c5') 40×40           c5'
      Conv(c5'→c6) 20×20          c6 (e.g., 1024) ← P5 output
      C2f(c6→c6') 20×20           c6'
      SPPF(c6'→c7) 20×20          c7

    The three outputs (P3, P4, P5) feed into the neck/FPN.
    """

    def __init__(self, width_multiple=0.50, depth_multiple=0.33, act="silu"):
        """
        Args:
            width_multiple: scales channel counts (0.25=nano, 0.50=small, 1.0=large)
            depth_multiple: scales number of C2f bottlenecks
        """
        super().__init__()

        # Base channels — scaled by width_multiple
        base_channels = {
            'c1': make_divisible(64 * width_multiple),
            'c2': make_divisible(128 * width_multiple),
            'c3': make_divisible(256 * width_multiple),
            'c4': make_divisible(512 * width_multiple),
            'c5': make_divisible(512 * width_multiple),  # Same as c4 before C2f
            'c6': make_divisible(1024 * width_multiple),
        }

        # Bottleneck counts — scaled by depth_multiple
        n = lambda base: max(1, int(base * depth_multiple))

        # --- Stem ---
        self.stem = nn.Sequential(
            Conv(3, base_channels['c1'], 3, 2),    # 640 → 320 (if 640 input)
            Conv(base_channels['c1'], base_channels['c2'], 3, 2),  # 320 → 160
        )

        # --- Stage 3 ---
        self.stage3 = nn.Sequential(
            C2f(base_channels['c2'], base_channels['c3'], n(3), shortcut=True),
            Conv(base_channels['c3'], base_channels['c4'], 3, 2),  # 160 → 80
        )

        # --- Stage 4 (P3 output) ---
        self.stage4 = nn.Sequential(
            C2f(base_channels['c4'], base_channels['c5'], n(6), shortcut=True),
            Conv(base_channels['c5'], base_channels['c5'], 3, 2),  # 80 → 40
        )

        # --- Stage 5 (P4 output) ---
        self.stage5 = nn.Sequential(
            C2f(base_channels['c5'], base_channels['c5'], n(6), shortcut=True),
            Conv(base_channels['c5'], base_channels['c6'], 3, 2),  # 40 → 20
        )

        # --- Stage 6 (P5 output) ---
        self.stage6 = nn.Sequential(
            C2f(base_channels['c6'], base_channels['c6'], n(3), shortcut=True),
            SPPF(base_channels['c6'], base_channels['c6'], kernel_size=5),
        )

        # Save output channel dimensions for the neck
        self.out_channels = [base_channels['c4'], base_channels['c5'], base_channels['c6']]

    def forward(self, x):
        """
        Returns:
            List of 3 feature maps: [P3, P4, P5]
            For 640×640 input: 80×80, 40×40, 20×20
        """
        x = self.stem(x)        # [B, c2, 160, 160]
        x = self.stage3(x)      # [B, c4, 80, 80]
        p3 = self.stage4[0](x)  # [B, c5, 80, 80] — before downsampling
        p4_in = self.stage4[1](p3)  # downsample
        p4 = self.stage5[0](p4_in)  # [B, c5, 40, 40] — before downsampling
        p5_in = self.stage5[1](p4)  # downsample
        p5 = self.stage6(p5_in)     # [B, c6, 20, 20]

        return [p3, p4, p5]


# =============================================================================
# Alternative backbones you can plug in for experiments
# =============================================================================

class EfficientNetBackbone(nn.Module):
    """
    Placeholder for EfficientNet backbone — lighter and often more accurate.
    TODO: Implement this as an accuracy improvement experiment.
    """
    def __init__(self, variant='efficientnet-b0', pretrained=True):
        super().__init__()
        # Import would go here
        raise NotImplementedError("EfficientNet backbone coming soon!")


class ResNetBackbone(nn.Module):
    """
    Placeholder for ResNet-50/101 backbone — simpler, well-understood baseline.
    TODO: Implement as a comparison baseline.
    """
    def __init__(self, variant='resnet50', pretrained=True):
        super().__init__()
        raise NotImplementedError("ResNet backbone coming soon!")
