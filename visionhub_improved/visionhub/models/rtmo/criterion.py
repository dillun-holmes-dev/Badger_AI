"""
RTMOPose criterion.

This mirrors MMPose RTMO training while preserving this repo's target format:
  - SimOTA assignment with OKS-driven dynamic-k
  - Varifocal classification targets from assignment overlaps
  - IoU bbox loss plus raw bbox auxiliary L1
  - proxy keypoint OKS loss and visibility BCE
  - DCC maximum-likelihood coordinate-classification loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .assigner import simota_assign
from visionhub.misc.box_ops import box_cxcywh_to_xyxy
from visionhub.misc.keypoint_loss import resolve_sigmas


EPS = 1e-8


def varifocal_loss(pred_logits, target_score, alpha=0.75, gamma=2.0):
    """MMPose/MMDetection-style Varifocal loss with sum reduction."""
    label = (target_score > 1e-4).to(target_score)
    weight = alpha * pred_logits.sigmoid().pow(gamma) * (1 - label) + target_score
    pred_logits = pred_logits.clamp(min=-10.0, max=10.0)
    loss = F.binary_cross_entropy_with_logits(pred_logits, target_score, reduction="none") * weight
    loss = torch.where(torch.isfinite(loss), loss, torch.zeros_like(loss))
    return loss.sum()


def iou_loss_square(pred_xyxy, gt_xyxy, eps=1e-16):
    """Aligned IoU loss in MMPose RTMO's square mode: 1 - IoU^2."""
    px1, py1, px2, py2 = pred_xyxy.unbind(-1)
    gx1, gy1, gx2, gy2 = gt_xyxy.unbind(-1)

    ix1 = torch.maximum(px1, gx1)
    iy1 = torch.maximum(py1, gy1)
    ix2 = torch.minimum(px2, gx2)
    iy2 = torch.minimum(py2, gy2)
    inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)

    pred_area = (px2 - px1).clamp(min=0) * (py2 - py1).clamp(min=0)
    gt_area = (gx2 - gx1).clamp(min=0) * (gy2 - gy1).clamp(min=0)
    iou = (inter / (pred_area + gt_area - inter).clamp(min=eps)).clamp(min=eps)
    return 1.0 - iou.pow(2)


def oks_loss_rtmo(kpts_xy, gt_kpts_xy, gt_vis, gt_area, sigmas, eps=1e-8):
    """MMPose OKSLoss in linear mode, with invisible keypoints masked."""
    sigmas = sigmas.to(device=kpts_xy.device, dtype=kpts_xy.dtype)
    valid = (gt_vis > 0).to(kpts_xy.dtype)
    dist = torch.norm(kpts_xy - gt_kpts_xy, dim=-1)
    dist = dist / gt_area.pow(0.5).clamp(min=eps).unsqueeze(-1)
    dist = dist / (sigmas.view(1, -1) * 2.0).clamp(min=eps)
    oks = torch.exp(-dist.pow(2) / 2.0)
    oks = (oks * (valid / valid.shape[-1])).sum(dim=-1)
    loss = 1.0 - oks
    return torch.where(valid.sum(dim=-1) > 0, loss, loss.new_zeros(loss.shape))


def mle_cc_loss(outputs, targets, target_weight=None):
    """MMPose MLECCLoss in log mode, returning one loss per instance."""
    prob = 1.0
    for output, target in zip(outputs, targets):
        prob = prob * (output * target).sum(dim=-1)

    loss = -torch.log(prob + 1e-4)
    loss = torch.where(torch.isfinite(loss), loss, torch.zeros_like(loss))

    if target_weight is not None:
        for _ in range(loss.ndim - target_weight.ndim):
            target_weight = target_weight.unsqueeze(-1)
        loss = loss * target_weight

    return loss.flatten(1).mean(dim=1)


def bbox_xyxy_to_cs(boxes_xyxy, padding=1.25):
    center = (boxes_xyxy[:, :2] + boxes_xyxy[:, 2:]) * 0.5
    scale = (boxes_xyxy[:, 2:] - boxes_xyxy[:, :2]).clamp(min=1.0) * padding
    return torch.cat([center, scale], dim=-1)


class RTMOCriterion(nn.Module):
    def __init__(
        self,
        num_classes,
        num_body_points,
        sigmas=None,
        weight_dict=None,
        topk_candidates=10,
        center_radius=2.5,
        assign_cls_weight=1.0,
        assign_iou_weight=3.0,
        assign_oks_weight=3.0,
        assign_vis_weight=0.0,
        dynamic_k_indicator="oks",
        use_keypoints_for_center=True,
        overlaps_power=0.5,
        bbox_padding=1.25,
        use_aux_loss=True,
        use_dcc_loss=True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_body_points = num_body_points
        self.register_buffer(
            "sigmas",
            torch.tensor(resolve_sigmas(sigmas, num_body_points), dtype=torch.float32),
        )
        self.weight_dict = weight_dict or {
            "loss_cls": 1.0,
            "loss_bbox": 5.0,
            "loss_kpts": 30.0,
            "loss_vis": 1.0,
            "loss_mle": 1.0,
            "loss_bbox_aux": 1.0,
        }

        self.topk = topk_candidates
        self.center_radius = center_radius
        self.assign_cls_weight = assign_cls_weight
        self.assign_iou_weight = assign_iou_weight
        self.assign_oks_weight = assign_oks_weight
        self.assign_vis_weight = assign_vis_weight
        self.dynamic_k_indicator = dynamic_k_indicator
        self.use_keypoints_for_center = use_keypoints_for_center
        self.overlaps_power = overlaps_power
        self.bbox_padding = bbox_padding
        self.use_aux_loss = use_aux_loss
        self.use_dcc_loss = use_dcc_loss
        object.__setattr__(self, "_dcc_ref", None)

    @property
    def dcc(self):
        return self._dcc_ref

    def set_dcc(self, dcc):
        object.__setattr__(self, "_dcc_ref", dcc)
        return self

    def forward(self, outputs, targets):
        cls_logits = outputs["cls_logits"]
        bbox_pred = outputs["bbox_pred"]
        bbox_decoded = outputs["bbox_decoded"]
        kpts_proxy = outputs.get("kpts_proxy_decoded", outputs["kpts_decoded"])
        kpt_vis_raw = outputs["kpt_vis_raw"]
        kpts_vis = outputs["kpts_vis"]
        pose_vecs = outputs.get("pose_vecs", None)
        anchor_pts = outputs["anchor_pts"]
        strides = outputs["strides"]

        B, N, C = cls_logits.shape
        device = cls_logits.device
        K = self.num_body_points
        sigmas = self.sigmas.to(device=device, dtype=cls_logits.dtype)
        priors = torch.cat([anchor_pts, strides[:, None], strides[:, None]], dim=-1)

        total_cls = cls_logits.new_zeros(())
        total_bbox = cls_logits.new_zeros(())
        total_kpts = cls_logits.new_zeros(())
        total_vis = cls_logits.new_zeros(())
        total_mle = cls_logits.new_zeros(())
        total_bbox_aux = cls_logits.new_zeros(())
        num_fg = 0
        dcc_pose_vecs = []
        dcc_bbox_cs = []
        dcc_anchor_pts = []
        dcc_gt_kpts = []
        dcc_gt_vis = []
        dcc_gt_area = []

        for b_idx in range(B):
            tgt = targets[b_idx]
            gt_boxes = tgt["boxes"].to(device)
            gt_labels = tgt["labels"].to(device).long()
            gt_kpts = tgt["keypoints"].to(device)
            gt_area = tgt.get("area", None)

            img_size = tgt["size"].to(device)
            H_img, W_img = img_size[0], img_size[1]
            scale_box = torch.stack((W_img, H_img, W_img, H_img)).to(dtype=gt_boxes.dtype)
            gt_boxes = box_cxcywh_to_xyxy(gt_boxes * scale_box)

            scale_kpts = torch.stack((W_img, H_img)).to(dtype=gt_kpts.dtype)
            gt_kpts_xy = gt_kpts[:, : K * 2].view(-1, K, 2) * scale_kpts
            gt_kpts_vis_raw = gt_kpts[:, K * 2 :]
            gt_kpts_visible = (gt_kpts_vis_raw > 0).to(gt_kpts_xy.dtype)

            if gt_area is None:
                bw = (gt_boxes[:, 2] - gt_boxes[:, 0]).clamp(min=0)
                bh = (gt_boxes[:, 3] - gt_boxes[:, 1]).clamp(min=0)
                gt_area = bw * bh
            else:
                gt_area = gt_area.to(device) * (H_img * W_img)
            gt_area = gt_area.clamp(min=1.0)

            cls_scores = cls_logits[b_idx].detach().clamp(min=-1e4, max=1e4).sigmoid().sqrt()
            fg_mask, assigned_gt, assigned_cls, assigned_box, assigned_score = simota_assign(
                cls_scores,
                bbox_decoded[b_idx].detach(),
                priors,
                gt_boxes,
                gt_labels,
                gt_keypoints=gt_kpts_xy,
                gt_keypoints_visible=gt_kpts_visible,
                pred_keypoints=kpts_proxy[b_idx].detach(),
                pred_keypoints_visible=kpts_vis[b_idx].detach(),
                gt_areas=gt_area,
                sigmas=sigmas,
                topk_candidates=self.topk,
                center_radius=self.center_radius,
                lambda_cls=self.assign_cls_weight,
                lambda_iou=self.assign_iou_weight,
                lambda_oks=self.assign_oks_weight,
                lambda_vis=self.assign_vis_weight,
                dynamic_k_indicator=self.dynamic_k_indicator,
                use_keypoints_for_center=self.use_keypoints_for_center,
            )

            # assigned_gt.numel() is a Python-side tensor size — no GPU sync needed.
            num_fg += assigned_gt.numel()

            target_score = torch.zeros(N, C, device=device, dtype=cls_logits.dtype)
            if assigned_gt.numel() > 0:
                cls_target = assigned_score.to(target_score).pow(self.overlaps_power).clamp(min=1e-3)
                target_score[fg_mask, assigned_cls] = cls_target

            total_cls = total_cls + varifocal_loss(cls_logits[b_idx], target_score)
            if assigned_gt.numel() == 0:
                continue

            pred_box_fg = bbox_decoded[b_idx][fg_mask]
            total_bbox = total_bbox + iou_loss_square(pred_box_fg, assigned_box).sum()

            if self.use_aux_loss:
                pos_priors = priors[fg_mask]
                gt_center = (assigned_box[:, :2] + assigned_box[:, 2:]) * 0.5
                gt_wh = (assigned_box[:, 2:] - assigned_box[:, :2]).clamp(min=EPS)
                stride_xy = pos_priors[:, 2:].clamp(min=EPS)
                bbox_aux_target = torch.cat(
                    [(gt_center - pos_priors[:, :2]) / stride_xy, torch.log(gt_wh / stride_xy + EPS)],
                    dim=-1,
                )
                total_bbox_aux = total_bbox_aux + F.l1_loss(
                    bbox_pred[b_idx][fg_mask], bbox_aux_target, reduction="sum"
                )

            pred_kpts_fg = kpts_proxy[b_idx][fg_mask]
            pred_vis_fg = kpt_vis_raw[b_idx][fg_mask]
            gt_kpts_fg_xy = gt_kpts_xy[assigned_gt]
            gt_kpts_fg_vis = gt_kpts_visible[assigned_gt]
            gt_area_fg = gt_area[assigned_gt]

            total_kpts = total_kpts + oks_loss_rtmo(
                pred_kpts_fg, gt_kpts_fg_xy, gt_kpts_fg_vis, gt_area_fg, sigmas
            ).sum()
            total_vis = total_vis + F.binary_cross_entropy_with_logits(
                pred_vis_fg, gt_kpts_fg_vis, reduction="sum"
            ) / max(K, 1)

            if self.use_dcc_loss and self.dcc is not None and pose_vecs is not None:
                dcc_pose_vecs.append(pose_vecs[b_idx][fg_mask])
                dcc_bbox_cs.append(bbox_xyxy_to_cs(pred_box_fg, padding=self.bbox_padding))
                dcc_anchor_pts.append(anchor_pts[fg_mask])
                dcc_gt_kpts.append(gt_kpts_fg_xy)
                dcc_gt_vis.append(gt_kpts_fg_vis)
                dcc_gt_area.append(gt_area_fg)

        if self.use_dcc_loss and self.dcc is not None and pose_vecs is not None and dcc_pose_vecs:
            dcc_param = next(self.dcc.parameters(), None)
            dcc_dtype = dcc_param.dtype if dcc_param is not None else pose_vecs.dtype
            pose_vecs_fg = torch.cat(dcc_pose_vecs, dim=0).to(dtype=dcc_dtype)
            bbox_cs = torch.cat(dcc_bbox_cs, dim=0).to(dtype=dcc_dtype)
            anchor_pts_fg = torch.cat(dcc_anchor_pts, dim=0).to(dtype=dcc_dtype)
            gt_kpts_fg_xy = torch.cat(dcc_gt_kpts, dim=0).to(dtype=dcc_dtype)
            gt_kpts_fg_vis = torch.cat(dcc_gt_vis, dim=0).to(dtype=dcc_dtype)
            gt_area_fg = torch.cat(dcc_gt_area, dim=0).to(dtype=dcc_dtype)

            _, pred_hms, dcc_sigmas = self.dcc.forward_train(
                pose_vecs_fg,
                bbox_cs,
                anchor_pts_fg,
            )
            target_hms = self.dcc.generate_target_heatmap(
                gt_kpts_fg_xy,
                bbox_cs,
                dcc_sigmas,
                gt_area_fg,
            )
            total_mle = total_mle + mle_cc_loss(
                pred_hms, target_hms, gt_kpts_fg_vis
            ).sum()

        normaliser = max(num_fg, 1)
        loss_dict = {
            "loss_cls": total_cls / normaliser * self.weight_dict.get("loss_cls", 1.0),
            "loss_bbox": total_bbox / normaliser * self.weight_dict.get("loss_bbox", 5.0),
            "loss_kpts": total_kpts / normaliser * self.weight_dict.get("loss_kpts", 30.0),
            "loss_vis": total_vis / normaliser * self.weight_dict.get("loss_vis", 1.0),
        }
        if self.use_dcc_loss:
            loss_dict["loss_mle"] = total_mle / normaliser * self.weight_dict.get("loss_mle", 1.0)
        if self.use_aux_loss:
            loss_dict["loss_bbox_aux"] = total_bbox_aux / normaliser * self.weight_dict.get("loss_bbox_aux", 1.0)
        return loss_dict
