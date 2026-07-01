"""
Standalone RTMO head, closely following MMPose's RTMOHeadModule.

It keeps this repo's public output structure while using the RTMO pieces that
matter for bbox/class/keypoint training:
  - split cls / pose feature streams
  - grouped pose branch shared by bbox, keypoint proxy, visibility, pose vectors
  - YOLO-style bbox decode: delta_xy, log_wh relative to point priors
  - proxy keypoint offsets decoded relative to point priors
  - Dynamic Coordinate Classification (DCC) for final keypoint coordinates
"""

import math
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..detrpose.hybrid_encoder import get_activation


EPS = 1e-8


class ConvBNAct(nn.Module):
    """ConvModule equivalent used by MMPose: conv + BN + activation."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        groups=1,
        act="silu",
        bn_momentum=0.03,
        bn_eps=0.001,
    ):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels, momentum=bn_momentum, eps=bn_eps)
        self.act = get_activation(act)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Scale(nn.Module):
    """Small learnable scalar used by DCC sigma prediction."""

    def __init__(self, init_value=1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(float(init_value)))

    def forward(self, x):
        return x * self.scale


class ScaleNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.scale = dim ** -0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1))

    def forward(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        return x / norm.clamp(min=self.eps) * self.g


class ChannelWiseScale(nn.Module):
    def __init__(self, dim, init_value=1.0):
        super().__init__()
        self.scale = nn.Parameter(init_value * torch.ones(dim))

    def forward(self, x):
        return x * self.scale


class GAUEncoder(nn.Module):
    """Gated Attention Unit used by MMPose DCC."""

    def __init__(
        self,
        in_token_dims,
        out_token_dims,
        expansion_factor=2,
        s=128,
        eps=1e-5,
        dropout_rate=0.0,
        act_fn="SiLU",
        bias=False,
        pos_enc="add",
    ):
        super().__init__()
        self.s = s
        self.e = int(in_token_dims * expansion_factor)
        self.pos_enc = pos_enc
        self.sqrt_s = math.sqrt(s)
        self.dropout_rate = dropout_rate

        self.ln = ScaleNorm(in_token_dims, eps=eps)
        self.uv = nn.Linear(in_token_dims, 2 * self.e + self.s, bias=bias)
        self.o = nn.Linear(self.e, out_token_dims, bias=bias)
        self.gamma = nn.Parameter(torch.rand((2, self.s)))
        self.beta = nn.Parameter(torch.rand((2, self.s)))
        self.act_fn = nn.SiLU(inplace=True) if act_fn.lower() in ("silu", "swish") else nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

        self.shortcut = in_token_dims == out_token_dims
        self.res_scale = ChannelWiseScale(in_token_dims) if self.shortcut else None
        nn.init.xavier_uniform_(self.uv.weight)

    def forward(self, x, mask=None, pos_enc=None):
        shortcut = x
        x = self.ln(x)
        uv = self.act_fn(self.uv(x))
        u, v, base = torch.split(uv, [self.e, self.e, self.s], dim=-1)

        dim = base.ndim - self.gamma.ndim + 1
        gamma = self.gamma.view(*((1,) * dim), *self.gamma.shape)
        beta = self.beta.view(*((1,) * dim), *self.beta.shape)
        base = base.unsqueeze(-2) * gamma + beta
        q, k = torch.unbind(base, dim=-2)

        if self.pos_enc == "add" and pos_enc is not None:
            pos_enc = pos_enc.reshape(*((1,) * (q.ndim - 2)), q.size(-2), q.size(-1))
            q = q + pos_enc
            k = k + pos_enc

        kernel = torch.square(F.relu(torch.matmul(q, k.transpose(-1, -2)) / self.sqrt_s))
        if mask is not None:
            kernel = kernel * mask
        kernel = self.dropout(kernel)
        out = self.o(u * torch.matmul(kernel, v))
        return self.res_scale(shortcut) + out if self.shortcut else out


class SinePositionalEncoding(nn.Module):
    """1-D sine/cosine encoding used for dynamic DCC bins."""

    def __init__(self, out_channels: int, temperature: float = 300.0):
        super().__init__()
        assert out_channels % 2 == 0
        pos_dim = out_channels // 2
        dim_t = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        self.register_buffer("dim_t", temperature ** dim_t)

    def forward(self, position):
        dim_t = self.dim_t.reshape(*((1,) * position.ndim), -1).to(position)
        freq = position.unsqueeze(-1) / dim_t
        return torch.cat((freq.cos(), freq.sin()), dim=-1)


class LinearBN1d(nn.Module):
    """Linear + BN1d with a safe one-sample fallback for small batches."""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)

    def forward(self, x):
        x = self.linear(x)
        if self.training and x.shape[0] == 1:
            return F.batch_norm(
                x,
                self.bn.running_mean,
                self.bn.running_var,
                self.bn.weight,
                self.bn.bias,
                training=False,
                momentum=self.bn.momentum,
                eps=self.bn.eps,
            )
        return self.bn(x)


class DCC(nn.Module):
    """Dynamic Coordinate Classifier from MMPose RTMO."""

    def __init__(
        self,
        in_channels: int,
        num_keypoints: int,
        feat_channels: int = 128,
        num_bins: Tuple[int, int] = (192, 256),
        spe_channels: int = 128,
        spe_temperature: float = 300.0,
        gau_cfg: Optional[dict] = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_keypoints = num_keypoints
        self.feat_channels = feat_channels
        self.num_bins = num_bins

        if gau_cfg is None:
            gau_cfg = {
                "s": 128,
                "expansion_factor": 2,
                "dropout_rate": 0.0,
                "act_fn": "SiLU",
                "pos_enc": "add",
            }

        self.spe = SinePositionalEncoding(spe_channels, temperature=spe_temperature)
        self.pose_to_kpts = LinearBN1d(in_channels, feat_channels * num_keypoints)
        self.x_fc = nn.Linear(spe_channels, feat_channels)
        self.y_fc = nn.Linear(spe_channels, feat_channels)
        self.sigma_fc = nn.Sequential(nn.Linear(in_channels, num_keypoints), nn.Sigmoid(), Scale(0.1))

        self.gau = GAUEncoder(
            in_token_dims=feat_channels,
            out_token_dims=feat_channels,
            expansion_factor=gau_cfg.get("expansion_factor", 2),
            s=gau_cfg.get("s", 128),
            dropout_rate=gau_cfg.get("dropout_rate", 0.0),
            act_fn=gau_cfg.get("act_fn", "SiLU"),
            pos_enc=gau_cfg.get("pos_enc", "add"),
        )
        if gau_cfg.get("pos_enc", "add") in ("add", "rope"):
            self.pos_enc = nn.Parameter(torch.randn(num_keypoints, gau_cfg.get("s", 128)))

        self.register_buffer("x_bins", torch.linspace(-0.5, 0.5, num_bins[0]))
        self.register_buffer("y_bins", torch.linspace(-0.5, 0.5, num_bins[1]))

    def _apply_softmax(self, x_hms, y_hms):
        x_hms = x_hms.clamp(min=-5e4, max=5e4)
        y_hms = y_hms.clamp(min=-5e4, max=5e4)
        pred_x = x_hms - x_hms.max(dim=-1, keepdim=True).values.detach()
        pred_y = y_hms - y_hms.max(dim=-1, keepdim=True).values.detach()
        prob_x = pred_x.exp()
        prob_y = pred_y.exp()
        prob_x = prob_x / (prob_x.sum(dim=-1, keepdim=True) + EPS)
        prob_y = prob_y / (prob_y.sum(dim=-1, keepdim=True) + EPS)
        return prob_x, prob_y

    def _get_bin_enc(self, bbox_cs, grids):
        center, scale = bbox_cs.split(2, dim=-1)
        center = center - grids
        x_bins = self.x_bins.view(*((1,) * (scale.ndim - 1)), -1).to(scale)
        y_bins = self.y_bins.view(*((1,) * (scale.ndim - 1)), -1).to(scale)
        x_bins = x_bins * scale[..., 0:1] + center[..., 0:1]
        y_bins = y_bins * scale[..., 1:2] + center[..., 1:2]
        return self.x_fc(self.spe(x_bins)), self.y_fc(self.spe(y_bins))

    def _pose_feats_to_heatmaps(self, pose_feats, x_bins_enc, y_bins_enc):
        kpt_feats = self.pose_to_kpts(pose_feats)
        kpt_feats = kpt_feats.reshape(*kpt_feats.shape[:-1], self.num_keypoints, self.feat_channels)
        kpt_feats = self.gau(kpt_feats, pos_enc=getattr(self, "pos_enc", None))
        x_hms = torch.matmul(kpt_feats, x_bins_enc.transpose(-1, -2).contiguous())
        y_hms = torch.matmul(kpt_feats, y_bins_enc.transpose(-1, -2).contiguous())
        return x_hms, y_hms

    def _decode_xy_heatmaps(self, x_hms, y_hms, bbox_cs):
        center, scale = bbox_cs.split(2, dim=-1)
        x_bins = self.x_bins.view(*((1,) * (scale.ndim - 1)), -1).to(scale)
        y_bins = self.y_bins.view(*((1,) * (scale.ndim - 1)), -1).to(scale)
        x_bins = x_bins * scale[..., 0:1] + center[..., 0:1]
        y_bins = y_bins * scale[..., 1:2] + center[..., 1:2]
        x = (x_hms * x_bins.unsqueeze(1)).sum(dim=-1)
        y = (y_hms * y_bins.unsqueeze(1)).sum(dim=-1)
        return torch.stack((x, y), dim=-1)

    def generate_target_heatmap(self, kpt_targets, bbox_cs, sigmas, areas):
        center, scale = bbox_cs.split(2, dim=-1)
        x_bins = self.x_bins.view(*((1,) * (scale.ndim - 1)), -1).to(scale)
        y_bins = self.y_bins.view(*((1,) * (scale.ndim - 1)), -1).to(scale)
        x_bins = x_bins * scale[..., 0:1] + center[..., 0:1]
        y_bins = y_bins * scale[..., 1:2] + center[..., 1:2]

        dist_x = torch.abs(kpt_targets[..., 0:1] - x_bins.unsqueeze(1))
        dist_y = torch.abs(kpt_targets[..., 1:2] - y_bins.unsqueeze(1))
        areas = areas.pow(0.5).clamp(min=1.0).reshape(-1, 1, 1)
        sigmas = sigmas.clamp(min=1e-3).unsqueeze(2)
        dist_x = dist_x / areas / sigmas
        dist_y = dist_y / areas / sigmas
        # Generate Gaussian distributions and normalize to sum to 1 (proper probability distributions)
        target_x = torch.exp(-dist_x.pow(2) / 2)
        target_y = torch.exp(-dist_y.pow(2) / 2)
        target_x = target_x / (target_x.sum(dim=-1, keepdim=True) + EPS)
        target_y = target_y / (target_y.sum(dim=-1, keepdim=True) + EPS)
        return target_x, target_y

    def forward_train(self, pose_feats, bbox_cs, grids):
        sigmas = self.sigma_fc(pose_feats)
        x_bins_enc, y_bins_enc = self._get_bin_enc(bbox_cs, grids)
        x_hms, y_hms = self._pose_feats_to_heatmaps(pose_feats, x_bins_enc, y_bins_enc)
        x_hms, y_hms = self._apply_softmax(x_hms, y_hms)
        pose_preds = self._decode_xy_heatmaps(x_hms, y_hms, bbox_cs)
        return pose_preds, (x_hms, y_hms), sigmas

    @torch.no_grad()
    def forward_test(self, pose_feats, bbox_cs, grids):
        x_bins_enc, y_bins_enc = self._get_bin_enc(bbox_cs, grids)
        x_hms, y_hms = self._pose_feats_to_heatmaps(pose_feats, x_bins_enc, y_bins_enc)
        x_hms, y_hms = self._apply_softmax(x_hms, y_hms)
        return self._decode_xy_heatmaps(x_hms, y_hms, bbox_cs)


def _valid_group_count(channels, preferred_groups):
    groups = math.gcd(int(channels), int(preferred_groups))
    return max(groups, 1)


class RTMOHead(nn.Module):
    """MMPose-style RTMO head with this repo's flattened output contract."""

    def __init__(
        self,
        in_channels=256,
        num_classes=80,
        num_body_points=17,
        feat_strides: Sequence[int] = (8, 16, 32),
        num_convs=2,
        act="silu",
        cls_feat_channels=None,
        num_groups=8,
        channels_per_group=36,
        pose_vec_channels=None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_body_points = num_body_points
        self.feat_strides = list(feat_strides)
        self.num_convs = num_convs

        branch_in = in_channels // 2
        cls_feat_channels = int(cls_feat_channels or in_channels)
        pose_out_channels = int(num_groups * channels_per_group)
        pose_vec_channels = int(pose_vec_channels or in_channels)
        num_levels = len(feat_strides)

        self.cls_branches = nn.ModuleList()
        self.pose_branches = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.bbox_preds = nn.ModuleList()
        self.kpt_reg_preds = nn.ModuleList()
        self.kpt_vis_preds = nn.ModuleList()
        self.pose_preds = nn.ModuleList()

        for _ in range(num_levels):
            cls_layers = []
            for i in range(num_convs):
                cls_layers.append(
                    ConvBNAct(
                        branch_in if i == 0 else cls_feat_channels,
                        cls_feat_channels,
                        act=act,
                    )
                )
            self.cls_branches.append(nn.Sequential(*cls_layers))
            self.cls_preds.append(nn.Conv2d(cls_feat_channels, num_classes, 1))

            pose_layers = []
            for i in range(num_convs * 2):
                in_ch = branch_in if i == 0 else pose_out_channels
                groups = 1 if i == 0 else _valid_group_count(pose_out_channels, num_groups)
                pose_layers.append(ConvBNAct(in_ch, pose_out_channels, groups=groups, act=act))
            self.pose_branches.append(nn.Sequential(*pose_layers))
            self.bbox_preds.append(nn.Conv2d(pose_out_channels, 4, 1))
            self.kpt_reg_preds.append(nn.Conv2d(pose_out_channels, num_body_points * 2, 1))
            self.kpt_vis_preds.append(nn.Conv2d(pose_out_channels, num_body_points, 1))
            self.pose_preds.append(nn.Conv2d(pose_out_channels, pose_vec_channels, 1))

        self._init_weights()

    def _init_weights(self):
        for pred, stride in zip(self.cls_preds, self.feat_strides):
            bias_cls = math.log(5.0 / max(self.num_classes, 1) / (640.0 / float(stride)) ** 2)
            nn.init.constant_(pred.bias, bias_cls)
            nn.init.normal_(pred.weight, std=0.01)
        for preds in (self.bbox_preds, self.kpt_reg_preds, self.kpt_vis_preds, self.pose_preds):
            for pred in preds:
                nn.init.normal_(pred.weight, std=0.01)
                if pred.bias is not None:
                    nn.init.constant_(pred.bias, 0.0)

    def forward(self, feats):
        cls_list, bbox_list, kpt_list, pose_vec_list = [], [], [], []
        anchor_pts_list, stride_list = [], []

        for i, (feat, stride) in enumerate(zip(feats, self.feat_strides)):
            B, _, H, W = feat.shape
            ys, xs = torch.meshgrid(
                torch.arange(H, device=feat.device, dtype=feat.dtype) + 0.5,
                torch.arange(W, device=feat.device, dtype=feat.dtype) + 0.5,
                indexing="ij",
            )
            anchors = torch.stack([xs, ys], dim=-1).reshape(-1, 2) * stride
            anchor_pts_list.append(anchors)
            stride_list.append(torch.full((H * W,), stride, device=feat.device, dtype=feat.dtype))

            cls_input, pose_input = feat.split(feat.size(1) // 2, dim=1)
            cls_feat = self.cls_branches[i](cls_input)
            pose_feat = self.pose_branches[i](pose_input)

            cls = self.cls_preds[i](cls_feat)
            bbox = self.bbox_preds[i](pose_feat)
            kpt_reg = self.kpt_reg_preds[i](pose_feat)
            kpt_vis = self.kpt_vis_preds[i](pose_feat)
            pose_vec = self.pose_preds[i](pose_feat)
            kpt = torch.cat([kpt_reg, kpt_vis], dim=1)

            cls_list.append(cls.permute(0, 2, 3, 1).reshape(B, -1, self.num_classes))
            bbox_list.append(bbox.permute(0, 2, 3, 1).reshape(B, -1, 4))
            kpt_list.append(kpt.permute(0, 2, 3, 1).reshape(B, -1, self.num_body_points * 3))
            pose_vec_list.append(pose_vec.permute(0, 2, 3, 1).reshape(B, -1, pose_vec.shape[1]))

        return (
            torch.cat(cls_list, dim=1),
            torch.cat(bbox_list, dim=1),
            torch.cat(kpt_list, dim=1),
            torch.cat(anchor_pts_list, dim=0),
            torch.cat(stride_list, dim=0),
            torch.cat(pose_vec_list, dim=1),
        )

    @staticmethod
    def decode_bbox(bbox_pred, anchor_points, strides):
        """Decode MMPose/YOLOX bbox deltas to absolute xyxy boxes."""
        squeeze = bbox_pred.dim() == 2
        if squeeze:
            bbox_pred = bbox_pred.unsqueeze(0)

        strides = strides.to(dtype=bbox_pred.dtype, device=bbox_pred.device).view(1, -1, 1)
        priors = anchor_points.to(dtype=bbox_pred.dtype, device=bbox_pred.device).view(1, -1, 2)
        xy = bbox_pred[..., :2] * strides + priors
        wh = bbox_pred[..., 2:].clamp(min=-12.0, max=12.0).exp() * strides
        decoded = torch.stack(
            [
                xy[..., 0] - wh[..., 0] / 2,
                xy[..., 1] - wh[..., 1] / 2,
                xy[..., 0] + wh[..., 0] / 2,
                xy[..., 1] + wh[..., 1] / 2,
            ],
            dim=-1,
        )
        return decoded.squeeze(0) if squeeze else decoded

    @staticmethod
    def decode_kpt_reg(kpt_offsets, anchor_points, strides):
        squeeze = kpt_offsets.dim() == 2
        if squeeze:
            kpt_offsets = kpt_offsets.unsqueeze(0)

        num_kpts = kpt_offsets.shape[-1] // 2
        offsets = kpt_offsets.reshape(*kpt_offsets.shape[:-1], num_kpts, 2)
        strides = strides.to(dtype=kpt_offsets.dtype, device=kpt_offsets.device).view(1, -1, 1, 1)
        priors = anchor_points.to(dtype=kpt_offsets.dtype, device=kpt_offsets.device).view(1, -1, 1, 2)
        decoded = offsets * strides + priors
        return decoded.squeeze(0) if squeeze else decoded

    @staticmethod
    def decode_keypoints(kpt_pred, anchor_points, strides):
        num_kpts = kpt_pred.shape[-1] // 3
        kpt_offsets = kpt_pred[..., : num_kpts * 2]
        vis_logits = kpt_pred[..., num_kpts * 2 :]
        return RTMOHead.decode_kpt_reg(kpt_offsets, anchor_points, strides), vis_logits.sigmoid()
