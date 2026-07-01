"""Detection-only RTMDet model definitions."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .head import ConvBNAct, RTMDetPoseHead
from .neck import PAFPN


class RTMDetDetectionHead(nn.Module):
    def __init__(
        self,
        in_channels: int = 256,
        num_classes: int = 80,
        num_body_points: int = 0,
        feat_strides: tuple[int, ...] = (8, 16, 32),
        num_convs: int = 2,
        act: str = "silu",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_body_points = int(num_body_points)
        self.feat_strides = list(feat_strides)
        self.num_convs = num_convs

        num_levels = len(feat_strides)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for _ in range(num_levels):
            self.cls_convs.append(
                nn.Sequential(*[ConvBNAct(in_channels, in_channels, 3, act=act) for _ in range(num_convs)])
            )
            self.reg_convs.append(
                nn.Sequential(*[ConvBNAct(in_channels, in_channels, 3, act=act) for _ in range(num_convs)])
            )

        self.cls_preds = nn.ModuleList([nn.Conv2d(in_channels, num_classes, 1) for _ in range(num_levels)])
        self.ltrb_preds = nn.ModuleList([nn.Conv2d(in_channels, 4, 1) for _ in range(num_levels)])

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, mean=0, std=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        for pred, stride in zip(self.cls_preds, self.feat_strides):
            bias_cls = math.log(5.0 / max(self.num_classes, 1) / (640.0 / float(stride)) ** 2)
            nn.init.constant_(pred.bias, bias_cls)
        for pred in self.ltrb_preds:
            nn.init.constant_(pred.bias, 1.0)

    def forward(self, feats: list[torch.Tensor]):
        all_cls, all_ltrb = [], []
        all_anchors, all_strides = [], []

        for level, (feat, stride) in enumerate(zip(feats, self.feat_strides)):
            batch_size, _, feat_h, feat_w = feat.shape

            cls_feat = self.cls_convs[level](feat)
            reg_feat = self.reg_convs[level](feat)

            cls_out = self.cls_preds[level](cls_feat)
            ltrb_out = self.ltrb_preds[level](reg_feat)

            all_cls.append(cls_out.permute(0, 2, 3, 1).reshape(batch_size, -1, self.num_classes))
            all_ltrb.append(ltrb_out.permute(0, 2, 3, 1).reshape(batch_size, -1, 4))

            anchors, strides = RTMDetPoseHead._make_anchor_grid(feat_h, feat_w, stride, feat.device)
            all_anchors.append(anchors)
            all_strides.append(strides)

        return (
            torch.cat(all_cls, dim=1),
            torch.cat(all_ltrb, dim=1),
            torch.cat(all_anchors, dim=0),
            torch.cat(all_strides, dim=0),
        )


class RTMDetDetection(nn.Module):
    def __init__(
        self,
        backbone,
        neck_out_channels: int = 256,
        neck_depth_mult: float = 1.0,
        num_classes: int = 80,
        num_body_points: int = 0,
        feat_strides: tuple[int, ...] = (8, 16, 32),
        head_num_convs: int = 2,
        act: str = "silu",
        post_processor=None,
    ):
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.num_body_points = int(num_body_points)
        self.feat_strides = list(feat_strides)

        self.neck = PAFPN(
            in_channels=backbone.num_channels,
            out_channels=neck_out_channels,
            depth_mult=neck_depth_mult,
            act=act,
        )
        self.head = RTMDetDetectionHead(
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

        cls_logits, ltrb_raw, anchor_pts, strides = self.head(neck_feats)
        batch_size = cls_logits.shape[0]
        bbox_decoded = RTMDetPoseHead.decode_ltrb(ltrb_raw, anchor_pts, strides)

        return {
            "cls_logits": cls_logits,
            "bbox_pred": ltrb_raw,
            "anchor_pts": anchor_pts,
            "strides": strides,
            "image_size": torch._shape_as_tensor(x)[-2:]
            .to(device=x.device, dtype=x.dtype)
            .unsqueeze(0)
            .expand(batch_size, 2),
            "bbox_decoded": bbox_decoded,
        }
