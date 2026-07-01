"""
RTMDet-Pose: single-stage anchor-free multi-person pose estimation.

Based on RTMDet (OpenMMLab, Apache 2.0). Key features vs RTMO:
  - No DCC component (simpler, faster)
  - LTRB distance-based bbox (l, t, r, b from anchor to box edges)
  - Keypoint offsets decoded via tanh × stride × range
"""

import torch
import torch.nn as nn

from .neck import PAFPN
from .head import RTMDetPoseHead


class RTMDetPose(nn.Module):
    """
    RTMDet-Pose single-stage pose estimation model.

    Args:
        backbone:          nn.Module – HGNetV2 backbone.
        neck_out_channels: PAFPN output channels (256).
        neck_depth_mult:   PAFPN depth multiplier.
        num_classes:       Number of object classes.
        num_body_points:   Keypoints per instance.
        feat_strides:      Head feature strides, usually [8, 16, 32].
        head_num_convs:    Conv depth per head branch.
        act:               Activation name ('silu', 'relu', …).
        post_processor:    Optional RTMDetPostProcess instance.
    """

    def __init__(
        self,
        backbone,
        neck_out_channels: int = 256,
        neck_depth_mult: float = 1.0,
        num_classes: int = 80,
        num_body_points: int = 17,
        feat_strides: tuple = (8, 16, 32),
        head_num_convs: int = 2,
        act: str = "silu",
        post_processor=None,
    ):
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.num_body_points = num_body_points
        self.feat_strides = list(feat_strides)

        backbone_channels = backbone.num_channels  # e.g. [256, 512, 1024] for B0

        self.neck = PAFPN(
            in_channels=backbone_channels,
            out_channels=neck_out_channels,
            depth_mult=neck_depth_mult,
            act=act,
        )

        self.head = RTMDetPoseHead(
            in_channels=neck_out_channels,
            num_classes=num_classes,
            num_body_points=num_body_points,
            feat_strides=feat_strides,
            num_convs=head_num_convs,
            act=act,
        )

        self.post_processor = post_processor

    def deploy(self):
        self.eval()
        for m in self.modules():
            if hasattr(m, "convert_to_deploy"):
                m.convert_to_deploy()
        return self

    # ──────────────────────────────────────────────────────────────────────────

    def forward(self, samples, targets=None):
        if hasattr(samples, "tensors"):
            x = samples.tensors
        else:
            x = samples

        # Backbone
        feats = self.backbone(x)

        # Neck
        neck_feats = self.neck(feats)
        if len(neck_feats) != len(self.feat_strides):
            neck_feats = neck_feats[-len(self.feat_strides):]

        # Head
        cls_logits, ltrb_raw, kpt_pred, anchor_pts, strides = self.head(neck_feats)

        # Decode
        B = cls_logits.shape[0]
        K = self.num_body_points

        bbox_decoded = RTMDetPoseHead.decode_ltrb(ltrb_raw, anchor_pts, strides)
        kpts_xy, kpts_vis = RTMDetPoseHead.decode_keypoints(kpt_pred, anchor_pts, strides)

        kpt_vis_raw = kpt_pred[..., K * 2:]   # [B, N, K]  raw logits for BCE loss

        outputs = {
            "cls_logits":   cls_logits,    # [B, N, C]
            "bbox_pred":    ltrb_raw,       # [B, N, 4]  raw LTRB (used for aux L1 loss)
            "kpt_pred":     kpt_pred,       # [B, N, K*3]
            "anchor_pts":   anchor_pts,     # [N, 2]
            "strides":      strides,        # [N]
            # Preserve image size as a tensor so ONNX export does not capture
            # Python scalars from shape values as constants.
            "image_size":   torch._shape_as_tensor(x)[-2:].to(device=x.device, dtype=x.dtype).unsqueeze(0).expand(B, 2),
            "bbox_decoded": bbox_decoded,   # [B, N, 4]  xyxy absolute
            "kpts_decoded": kpts_xy,        # [B, N, K, 2]
            "kpts_vis":     kpts_vis,       # [B, N, K]  sigmoid scores
            "kpt_vis_raw":  kpt_vis_raw,    # [B, N, K]  raw logits
        }

        return outputs
