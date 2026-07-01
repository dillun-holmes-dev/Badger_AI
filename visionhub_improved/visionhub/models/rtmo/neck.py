"""
RTMOPose Neck: Path Aggregation Feature Pyramid Network (PAFPN)
Inspired by RTMDet / RTMO (Apache 2.0, OpenMMLab)
Standalone PyTorch - no MMDet dependency.
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from ..detrpose.hybrid_encoder import (
    ConvNormLayer, ConvNormLayer_fuse, get_activation, SCDown, CSPLayer, VGGBlock
)


class RepNCSPELAN4(nn.Module):
    """Efficient local attention block reused from hybrid_encoder pattern."""
    def __init__(self, c1, c2, c3, c4, n=3, bias=False, act="silu"):
        super().__init__()
        self.c = c3 // 2
        self.cv1 = ConvNormLayer_fuse(c1, c3, 1, 1, bias=bias, act=act)
        self.cv2 = nn.Sequential(
            CSPLayer(c3 // 2, c4, n, 1, bias=bias, act=act, bottletype=VGGBlock),
            ConvNormLayer_fuse(c4, c4, 3, 1, bias=bias, act=act)
        )
        self.cv3 = nn.Sequential(
            CSPLayer(c4, c4, n, 1, bias=bias, act=act, bottletype=VGGBlock),
            ConvNormLayer_fuse(c4, c4, 3, 1, bias=bias, act=act)
        )
        self.cv4 = ConvNormLayer_fuse(c3 + (2 * c4), c2, 1, 1, bias=bias, act=act)

    def forward(self, x):
        y0, y1 = self.cv1(x).split((self.c, self.c), 1)
        cv2_out = self.cv2(y1)
        cv3_out = self.cv3(cv2_out)
        return self.cv4(torch.cat([y0, y1, cv2_out, cv3_out], 1))


class PAFPN(nn.Module):
    """
    Path Aggregation Feature Pyramid Network.

    Args:
        in_channels:  List of backbone output channels, e.g. [256, 512, 1024]
        out_channels: Unified output channels (256)
        depth_mult:   Scale depth of CSP blocks
        act:          Activation function name
    """

    def __init__(
        self,
        in_channels=(256, 512, 1024),
        out_channels=256,
        depth_mult=1.0,
        act="silu",
    ):
        super().__init__()
        n_levels = len(in_channels)

        # Project each backbone level to out_channels
        self.input_proj = nn.ModuleList([
            nn.Sequential(OrderedDict([
                ("conv", nn.Conv2d(c, out_channels, 1, bias=False)),
                ("norm", nn.BatchNorm2d(out_channels)),
            ]))
            for c in in_channels
        ])

        # Top-down FPN
        self.fpn_laterals = nn.ModuleList([
            ConvNormLayer(out_channels, out_channels, 1, 1, act=act)
            for _ in range(n_levels - 1)
        ])
        self.fpn_blocks = nn.ModuleList([
            RepNCSPELAN4(
                out_channels * 2, out_channels, out_channels * 2,
                round(out_channels // 2), round(3 * depth_mult), act=act
            )
            for _ in range(n_levels - 1)
        ])

        # Bottom-up PAN
        self.pan_downsamples = nn.ModuleList([
            SCDown(out_channels, out_channels, 3, 2)
            for _ in range(n_levels - 1)
        ])
        self.pan_blocks = nn.ModuleList([
            RepNCSPELAN4(
                out_channels * 2, out_channels, out_channels * 2,
                round(out_channels // 2), round(3 * depth_mult), act=act
            )
            for _ in range(n_levels - 1)
        ])

        self._init_weights()

    def _init_weights(self):
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)

    def forward(self, feats):
        """
        Args:
            feats: list of [B, Ci, Hi, Wi] backbone feature maps (low→high resolution)
        Returns:
            outs: list of [B, out_channels, Hi, Wi] (same order, low→high resolution)
        """
        proj = [self.input_proj[i](f) for i, f in enumerate(feats)]

        # Top-down path: highest level → lowest
        inner = [proj[-1]]
        for idx in range(len(proj) - 1, 0, -1):
            fpn_idx = len(proj) - 1 - idx          # 0, 1, ...
            lat = self.fpn_laterals[fpn_idx](inner[0])
            up = F.interpolate(lat, scale_factor=2.0, mode="nearest")
            fused = self.fpn_blocks[fpn_idx](torch.cat([up, proj[idx - 1]], dim=1))
            inner.insert(0, fused)

        # Bottom-up path
        outs = [inner[0]]
        for i in range(len(inner) - 1):
            down = self.pan_downsamples[i](outs[-1])
            fused = self.pan_blocks[i](torch.cat([down, inner[i + 1]], dim=1))
            outs.append(fused)

        return outs
