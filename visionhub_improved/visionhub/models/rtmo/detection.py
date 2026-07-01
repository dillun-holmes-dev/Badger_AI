"""Detection-only RTMO model definitions."""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn

from .head import ConvBNAct, RTMOHead, _valid_group_count
from .neck import PAFPN


class RTMODetectionHead(nn.Module):
    def __init__(
        self,
        in_channels: int = 256,
        num_classes: int = 80,
        num_body_points: int = 0,
        feat_strides: Sequence[int] = (8, 16, 32),
        num_convs: int = 2,
        act: str = "silu",
        cls_feat_channels=None,
        num_groups: int = 8,
        channels_per_group: int = 36,
        pose_vec_channels=None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_body_points = int(num_body_points)
        self.feat_strides = list(feat_strides)
        self.num_convs = num_convs

        branch_in = in_channels // 2
        cls_feat_channels = int(cls_feat_channels or in_channels)
        det_out_channels = int(num_groups * channels_per_group)
        num_levels = len(feat_strides)

        self.cls_branches = nn.ModuleList()
        self.det_branches = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.bbox_preds = nn.ModuleList()

        for _ in range(num_levels):
            cls_layers = []
            for idx in range(num_convs):
                cls_layers.append(
                    ConvBNAct(
                        branch_in if idx == 0 else cls_feat_channels,
                        cls_feat_channels,
                        act=act,
                    )
                )
            self.cls_branches.append(nn.Sequential(*cls_layers))
            self.cls_preds.append(nn.Conv2d(cls_feat_channels, num_classes, 1))

            det_layers = []
            for idx in range(num_convs * 2):
                in_ch = branch_in if idx == 0 else det_out_channels
                groups = 1 if idx == 0 else _valid_group_count(det_out_channels, num_groups)
                det_layers.append(ConvBNAct(in_ch, det_out_channels, groups=groups, act=act))
            self.det_branches.append(nn.Sequential(*det_layers))
            self.bbox_preds.append(nn.Conv2d(det_out_channels, 4, 1))

        self._init_weights()

    def _init_weights(self) -> None:
        for pred, stride in zip(self.cls_preds, self.feat_strides):
            bias_cls = math.log(5.0 / max(self.num_classes, 1) / (640.0 / float(stride)) ** 2)
            nn.init.constant_(pred.bias, bias_cls)
            nn.init.normal_(pred.weight, std=0.01)
        for pred in self.bbox_preds:
            nn.init.normal_(pred.weight, std=0.01)
            if pred.bias is not None:
                nn.init.constant_(pred.bias, 0.0)

    def forward(self, feats):
        cls_list, bbox_list = [], []
        anchor_pts_list, stride_list = [], []

        for idx, (feat, stride) in enumerate(zip(feats, self.feat_strides)):
            batch_size, _, feat_h, feat_w = feat.shape
            ys, xs = torch.meshgrid(
                torch.arange(feat_h, device=feat.device, dtype=feat.dtype) + 0.5,
                torch.arange(feat_w, device=feat.device, dtype=feat.dtype) + 0.5,
                indexing="ij",
            )
            anchors = torch.stack([xs, ys], dim=-1).reshape(-1, 2) * stride
            anchor_pts_list.append(anchors)
            stride_list.append(torch.full((feat_h * feat_w,), stride, device=feat.device, dtype=feat.dtype))

            cls_input, det_input = feat.split(feat.size(1) // 2, dim=1)
            cls_feat = self.cls_branches[idx](cls_input)
            det_feat = self.det_branches[idx](det_input)

            cls = self.cls_preds[idx](cls_feat)
            bbox = self.bbox_preds[idx](det_feat)

            cls_list.append(cls.permute(0, 2, 3, 1).reshape(batch_size, -1, self.num_classes))
            bbox_list.append(bbox.permute(0, 2, 3, 1).reshape(batch_size, -1, 4))

        return (
            torch.cat(cls_list, dim=1),
            torch.cat(bbox_list, dim=1),
            torch.cat(anchor_pts_list, dim=0),
            torch.cat(stride_list, dim=0),
        )


class RTMODetection(nn.Module):
    def __init__(
        self,
        backbone,
        neck_out_channels=256,
        neck_depth_mult=1.0,
        num_classes=80,
        num_body_points=0,
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
        self.num_body_points = int(num_body_points)
        self.feat_strides = list(feat_strides)
        self.bbox_padding = bbox_padding
        self.post_processor = post_processor

        self.neck = PAFPN(
            in_channels=backbone.num_channels,
            out_channels=neck_out_channels,
            depth_mult=neck_depth_mult,
            act=act,
        )
        self.head = RTMODetectionHead(
            in_channels=neck_out_channels,
            num_classes=num_classes,
            num_body_points=num_body_points,
            feat_strides=feat_strides,
            num_convs=head_num_convs,
            cls_feat_channels=head_cls_feat_channels or neck_out_channels,
            num_groups=head_num_groups,
            channels_per_group=head_channels_per_group,
            pose_vec_channels=head_pose_vec_channels or neck_out_channels,
            act=act,
        )

    def deploy(self):
        self.eval()
        for module in self.modules():
            if hasattr(module, "convert_to_deploy"):
                module.convert_to_deploy()
        return self

    def forward(self, samples, targets=None):
        x = samples.tensors if hasattr(samples, "tensors") else samples

        feats = self.backbone(x)
        neck_feats = self.neck(feats)
        if len(neck_feats) != len(self.feat_strides):
            neck_feats = neck_feats[-len(self.feat_strides) :]

        cls_logits, bbox_pred, anchor_pts, strides = self.head(neck_feats)
        batch_size = cls_logits.shape[0]
        bbox_decoded = RTMOHead.decode_bbox(bbox_pred, anchor_pts, strides)

        return {
            "cls_logits": cls_logits,
            "bbox_pred": bbox_pred,
            "anchor_pts": anchor_pts,
            "strides": strides,
            "image_size": torch._shape_as_tensor(x)[-2:]
            .to(device=x.device, dtype=x.dtype)
            .unsqueeze(0)
            .expand(batch_size, 2),
            "bbox_decoded": bbox_decoded,
        }
