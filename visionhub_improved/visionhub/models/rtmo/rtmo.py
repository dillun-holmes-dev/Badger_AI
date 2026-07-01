"""
RTMOPose: Real-time multi-person pose estimation.
Standalone PyTorch model with an MMPose-style RTMO head.
"""

import torch
import torch.nn as nn

from .neck import PAFPN
from .head import DCC, RTMOHead


class RTMOPose(nn.Module):
    """
    RTMOPose single-stage pose estimation model.

    Args:
        backbone:          nn.Module – HGNetV2 backbone (instantiated externally via config).
        neck_out_channels: Unified PAFPN output channels (256).
        neck_depth_mult:   Depth multiplier for PAFPN CSP blocks.
        num_classes:       Number of object classes.
        num_body_points:   Keypoints per instance.
        feat_strides:      Strides consumed by the RTMO head, usually [8, 16, 32].
        head_num_convs:    Conv depth per head branch.
        act:               Activation name ('silu', 'relu', …).
        post_processor:    Optional RTMOPostProcess instance (used in eval/export).
    """

    def __init__(
        self,
        backbone,
        neck_out_channels=256,
        neck_depth_mult=1.0,
        num_classes=80,
        num_body_points=17,
        feat_strides=(8, 16, 32),
        head_num_convs=2,
        head_cls_feat_channels=None,
        head_num_groups=8,
        head_channels_per_group=36,
        head_pose_vec_channels=None,
        dcc_feat_channels=128,
        dcc_num_bins=(192, 256),
        dcc_spe_channels=128,
        bbox_padding=1.25,
        act="silu",
        post_processor=None,
    ):
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.num_body_points = num_body_points
        self.feat_strides = list(feat_strides)
        self.bbox_padding = bbox_padding

        # Backbone output channels come from backbone.num_channels
        # (HGNetV2 returns len(return_idx) feature levels)
        backbone_channels = backbone.num_channels   # e.g. [256, 512, 1024] for B0

        self.neck = PAFPN(
            in_channels=backbone_channels,
            out_channels=neck_out_channels,
            depth_mult=neck_depth_mult,
            act=act,
        )

        head_pose_vec_channels = head_pose_vec_channels or neck_out_channels
        self.head = RTMOHead(
            in_channels=neck_out_channels,
            num_classes=num_classes,
            num_body_points=num_body_points,
            feat_strides=feat_strides,
            num_convs=head_num_convs,
            cls_feat_channels=head_cls_feat_channels or neck_out_channels,
            num_groups=head_num_groups,
            channels_per_group=head_channels_per_group,
            pose_vec_channels=head_pose_vec_channels,
            act=act,
        )

        self.dcc = DCC(
            in_channels=head_pose_vec_channels,
            num_keypoints=num_body_points,
            feat_channels=dcc_feat_channels,
            num_bins=tuple(dcc_num_bins),
            spe_channels=dcc_spe_channels,
        )

        self.post_processor = post_processor
        if self.post_processor is not None and hasattr(self.post_processor, "set_dcc"):
            self.post_processor.set_dcc(self.dcc)

    def deploy(self):
        self.eval()
        for m in self.modules():
            if hasattr(m, "convert_to_deploy"):
                m.convert_to_deploy()
        return self

    # ──────────────────────────────────────────────────────────────────────────
    def forward(self, samples, targets=None):
        """
        Args:
            samples:  NestedTensor or plain [B, 3, H, W] tensor.
            targets:  list of target dicts (used in training to pass to criterion).
                      Not consumed here – the criterion is called from the solver.
        Returns:
            dict with raw head outputs. The solver/inference wrapper owns
            post-processing so evaluation can use the true target sizes.
        """
        # Handle NestedTensor (same pattern as DETRPose)
        if hasattr(samples, "tensors"):
            x = samples.tensors
        else:
            x = samples

        # Backbone
        feats = self.backbone(x)   # list of feature maps [stage2, stage3, stage4]

        # Neck
        neck_feats = self.neck(feats)
        if len(neck_feats) != len(self.feat_strides):
            neck_feats = neck_feats[-len(self.feat_strides):]

        # Head
        cls_logits, bbox_pred, kpt_pred, anchor_pts, strides, pose_vecs = self.head(neck_feats)

        # Decode predictions for loss / postprocess
        # Do this once here so both criterion and postprocessor can share them
        bbox_decoded = self._decode_bbox_all(bbox_pred, anchor_pts, strides)
        kpts_xy, kpts_vis = self.head.decode_keypoints(kpt_pred, anchor_pts, strides)
        B = cls_logits.shape[0]
        N = cls_logits.shape[1]
        K = self.num_body_points

        # raw visibility logits for BCE loss
        kpt_vis_raw = kpt_pred[..., K * 2:]  # [B, N, K]
        kpt_offsets = kpt_pred[..., :K * 2]

        outputs = {
            "cls_logits":   cls_logits,    # [B, N, C]
            "bbox_pred":    bbox_pred,      # [B, N, 4]  raw delta_xy/log_wh
            "kpt_pred":     kpt_pred,       # [B, N, K*3] proxy xy offsets + vis
            "kpt_offsets":  kpt_offsets,    # [B, N, K*2]
            "anchor_pts":   anchor_pts,     # [N, 2]
            "strides":      strides,        # [N]
            # Keep image size tensor-derived so ONNX export preserves dynamic
            # input shapes instead of freezing Python shape constants.
            "image_size":   torch._shape_as_tensor(x)[-2:].to(device=x.device, dtype=x.dtype).unsqueeze(0).expand(B, 2),
            "bbox_decoded": bbox_decoded,   # [B, N, 4]  xyxy absolute
            "kpts_decoded": kpts_xy,        # [B, N, K, 2] proxy; postprocess may replace with DCC
            "kpts_proxy_decoded": kpts_xy,  # [B, N, K, 2] used for SimOTA/OKS proxy loss
            "kpts_vis":     kpts_vis,       # [B, N, K]  sigmoid scores
            "kpt_vis_raw":  kpt_vis_raw,    # [B, N, K]  raw logits
            "pose_vecs":    pose_vecs,      # [B, N, C]  DCC pose vectors
        }

        return outputs

    # ──────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _decode_bbox_all(bbox_pred, anchor_pts, strides):
        """
        Decode [B, N, 4] delta_xy/log_wh -> [B, N, 4] xyxy absolute.
        """
        return RTMOHead.decode_bbox(bbox_pred, anchor_pts, strides)
