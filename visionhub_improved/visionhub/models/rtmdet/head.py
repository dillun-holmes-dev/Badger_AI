"""
RTMDet-Pose Head.

Standalone PyTorch implementation of the RTMDet detection + keypoint head.
Key differences from RTMO:
  - No DCC component (simpler, faster inference)
  - LTRB distance-based bbox regression (l, t, r, b distances from anchor to box edges)
  - Keypoints as direct offset predictions (tanh-gated, stride-scaled)
  - Shared reg+kpt feature branch (same as original RTMDet)
"""

import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..detrpose.hybrid_encoder import get_activation


EPS = 1e-8
KPT_RANGE = 5.0  # keypoints can be at most KPT_RANGE strides from anchor


class ConvBNAct(nn.Module):
    """Conv + BN + activation."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 groups=1, act="silu", bn_momentum=0.03, bn_eps=0.001):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, momentum=bn_momentum, eps=bn_eps)
        self.act = get_activation(act)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class RTMDetPoseHead(nn.Module):
    """
    RTMDet-Pose detection head.

    For each feature level produces:
      - cls_logits  [B, num_classes, H, W]
      - ltrb_raw    [B, 4, H, W]   (raw LTRB distances, decoded via softplus × stride)
      - kpt_offsets [B, K*2, H, W] (raw offsets, decoded via tanh × stride × KPT_RANGE)
      - vis_logits  [B, K, H, W]   (raw visibility logits)

    Args:
        in_channels:  Neck output channels (same for all levels).
        num_classes:  Number of object categories.
        num_body_points: Keypoints per instance.
        feat_strides: Feature strides, e.g. [8, 16, 32].
        num_convs:    Stacked conv depth per branch.
        act:          Activation name.
    """

    def __init__(
        self,
        in_channels: int = 256,
        num_classes: int = 80,
        num_body_points: int = 17,
        feat_strides: Tuple[int, ...] = (8, 16, 32),
        num_convs: int = 2,
        act: str = "silu",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_body_points = num_body_points
        self.feat_strides = list(feat_strides)
        self.num_convs = num_convs
        K = num_body_points

        # Per-level branches (separate BN per level, shared conv weights)
        n_levels = len(feat_strides)

        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for _ in range(n_levels):
            cls_branch = nn.Sequential(*[
                ConvBNAct(in_channels, in_channels, 3, act=act)
                for _ in range(num_convs)
            ])
            reg_branch = nn.Sequential(*[
                ConvBNAct(in_channels, in_channels, 3, act=act)
                for _ in range(num_convs)
            ])
            self.cls_convs.append(cls_branch)
            self.reg_convs.append(reg_branch)

        # Prediction heads (1×1 convs, one set per level)
        self.cls_preds = nn.ModuleList([
            nn.Conv2d(in_channels, num_classes, 1) for _ in range(n_levels)
        ])
        self.ltrb_preds = nn.ModuleList([
            nn.Conv2d(in_channels, 4, 1) for _ in range(n_levels)
        ])
        self.kpt_preds = nn.ModuleList([
            nn.Conv2d(in_channels, K * 2 + K, 1) for _ in range(n_levels)  # offsets + vis
        ])

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        for pred, stride in zip(self.cls_preds, self.feat_strides):
            bias_cls = math.log(5.0 / max(self.num_classes, 1) / (640.0 / float(stride)) ** 2)
            nn.init.constant_(pred.bias, bias_cls)
        for pred in self.ltrb_preds:
            nn.init.constant_(pred.bias, 1.0)  # start with unit distances

    # ── Grid generation ──────────────────────────────────────────────────────

    @staticmethod
    def _make_anchor_grid(feat_h: int, feat_w: int, stride: int, device) -> Tuple:
        """Returns anchor points [H*W, 2] and stride tensor [H*W] for one level."""
        ys = (torch.arange(feat_h, device=device, dtype=torch.float32) + 0.5) * stride
        xs = (torch.arange(feat_w, device=device, dtype=torch.float32) + 0.5) * stride
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        anchors = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)   # [H*W, 2]
        strides = anchors.new_full((anchors.shape[0],), fill_value=stride)    # [H*W]
        return anchors, strides

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(self, feats: List[torch.Tensor]):
        """
        Args:
            feats: list of [B, C, H_i, W_i] neck outputs (one per stride level).
        Returns:
            cls_logits  [B, N, num_classes]
            ltrb_raw    [B, N, 4]
            kpt_pred    [B, N, K*3]   (K*2 offsets + K vis logits)
            anchor_pts  [N, 2]
            strides     [N]
        """
        all_cls, all_ltrb, all_kpt = [], [], []
        all_anchors, all_strides = [], []

        for lvl, (feat, stride) in enumerate(zip(feats, self.feat_strides)):
            B, _, H, W = feat.shape

            # Branches
            cls_feat = self.cls_convs[lvl](feat)
            reg_feat = self.reg_convs[lvl](feat)

            # Predictions
            cls_out  = self.cls_preds[lvl](cls_feat)   # [B, C, H, W]
            ltrb_out = self.ltrb_preds[lvl](reg_feat)  # [B, 4, H, W]
            kpt_out  = self.kpt_preds[lvl](reg_feat)   # [B, K*3, H, W]

            # Flatten spatial: [B, C, H, W] → [B, H*W, C]
            all_cls.append(cls_out.permute(0, 2, 3, 1).reshape(B, -1, self.num_classes))
            all_ltrb.append(ltrb_out.permute(0, 2, 3, 1).reshape(B, -1, 4))
            all_kpt.append(kpt_out.permute(0, 2, 3, 1).reshape(B, -1, self.num_body_points * 3))

            anchors, strides = self._make_anchor_grid(H, W, stride, feat.device)
            all_anchors.append(anchors)
            all_strides.append(strides)

        cls_logits = torch.cat(all_cls,  dim=1)   # [B, N, C]
        ltrb_raw   = torch.cat(all_ltrb, dim=1)   # [B, N, 4]
        kpt_pred   = torch.cat(all_kpt,  dim=1)   # [B, N, K*3]
        anchor_pts = torch.cat(all_anchors, dim=0)  # [N, 2]
        strides    = torch.cat(all_strides,  dim=0) # [N]

        return cls_logits, ltrb_raw, kpt_pred, anchor_pts, strides

    # ── Decode helpers ───────────────────────────────────────────────────────

    @staticmethod
    def decode_ltrb(ltrb_raw: torch.Tensor, anchor_pts: torch.Tensor,
                    strides: torch.Tensor) -> torch.Tensor:
        """
        Decode LTRB raw predictions to xyxy absolute coordinates.
        ltrb_raw: [B, N, 4]   — raw head output (unbounded)
        anchor_pts: [N, 2]    — (cx, cy) in absolute pixels
        strides:    [N]
        Returns: [B, N, 4]  xyxy absolute
        """
        # softplus ensures positive distances; scale by stride
        ltrb = F.softplus(ltrb_raw) * strides.unsqueeze(-1)  # [B, N, 4]
        cx = anchor_pts[:, 0]   # [N]
        cy = anchor_pts[:, 1]   # [N]
        x1 = cx - ltrb[..., 0]
        y1 = cy - ltrb[..., 1]
        x2 = cx + ltrb[..., 2]
        y2 = cy + ltrb[..., 3]
        return torch.stack([x1, y1, x2, y2], dim=-1)

    @staticmethod
    def decode_keypoints(kpt_pred: torch.Tensor, anchor_pts: torch.Tensor,
                         strides: torch.Tensor) -> Tuple:
        """
        Decode raw keypoint predictions to absolute xy + visibility.
        kpt_pred:   [B, N, K*3]  (K*2 offsets + K vis logits)
        anchor_pts: [N, 2]
        strides:    [N]
        Returns:
            kpt_xy:  [B, N, K, 2]  absolute pixel coordinates
            kpt_vis: [B, N, K]     visibility scores (sigmoid)
        """
        K = kpt_pred.shape[-1] // 3
        offsets  = kpt_pred[..., :K * 2].view(*kpt_pred.shape[:-1], K, 2)  # [B, N, K, 2]
        vis_raw  = kpt_pred[..., K * 2:]                                     # [B, N, K]

        # tanh → bounded ±1, then scale by stride × range
        scaled_offsets = torch.tanh(offsets) * strides[:, None, None] * KPT_RANGE
        kpt_xy = anchor_pts[:, :2].unsqueeze(1) + scaled_offsets  # broadcast [N,2] + [B,N,K,2]
        kpt_vis = vis_raw.sigmoid()
        return kpt_xy, kpt_vis
