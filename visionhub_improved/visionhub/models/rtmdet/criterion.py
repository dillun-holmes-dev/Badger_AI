"""
RTMDet-Pose criterion.

Losses:
  loss_cls      — Varifocal loss on class logits (SimOTA-assigned targets).
  loss_bbox     — IoU² loss on decoded xyxy boxes.
  loss_kpts     — OKS loss on decoded keypoint positions.
  loss_vis      — BCE on keypoint visibility logits.
  loss_bbox_aux — L1 loss on LTRB normalized to stride (regression auxiliary).

Assignment: SimOTA (reused from RTMO assigner — OKS-driven dynamic-k).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse well-tested loss functions from RTMO
from ..rtmo.criterion import varifocal_loss, iou_loss_square, oks_loss_rtmo
from ..rtmo.assigner import simota_assign
from visionhub.misc.box_ops import box_cxcywh_to_xyxy
from visionhub.misc.keypoint_loss import resolve_sigmas


EPS = 1e-8


class RTMDetCriterion(nn.Module):
    def __init__(
        self,
        num_classes: int,
        num_body_points: int,
        sigmas=None,
        weight_dict=None,
        topk_candidates: int = 10,
        center_radius: float = 2.5,
        assign_cls_weight: float = 1.0,
        assign_iou_weight: float = 3.0,
        assign_oks_weight: float = 3.0,
        use_keypoints_for_center: bool = True,
        dynamic_k_indicator: str = "oks",
        overlaps_power: float = 0.5,
        use_aux_loss: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_body_points = num_body_points
        self.register_buffer(
            "sigmas",
            torch.tensor(resolve_sigmas(sigmas, num_body_points), dtype=torch.float32),
        )
        self.weight_dict = weight_dict or {
            "loss_cls":      1.0,
            "loss_bbox":     5.0,
            "loss_kpts":     30.0,
            "loss_vis":      1.0,
            "loss_bbox_aux": 1.0,
        }
        self.topk = topk_candidates
        self.center_radius = center_radius
        self.assign_cls_weight = assign_cls_weight
        self.assign_iou_weight = assign_iou_weight
        self.assign_oks_weight = assign_oks_weight
        self.use_keypoints_for_center = use_keypoints_for_center
        self.dynamic_k_indicator = dynamic_k_indicator
        self.overlaps_power = overlaps_power
        self.use_aux_loss = use_aux_loss

    def forward(self, outputs, targets):
        cls_logits   = outputs["cls_logits"]    # [B, N, C]
        ltrb_raw     = outputs["bbox_pred"]      # [B, N, 4]  raw LTRB
        bbox_decoded = outputs["bbox_decoded"]   # [B, N, 4]  xyxy absolute
        kpts_xy      = outputs["kpts_decoded"]   # [B, N, K, 2]
        kpt_vis_raw  = outputs["kpt_vis_raw"]    # [B, N, K]
        kpts_vis     = outputs["kpts_vis"]       # [B, N, K]  sigmoid
        anchor_pts   = outputs["anchor_pts"]     # [N, 2]
        strides      = outputs["strides"]        # [N]

        B, N, C = cls_logits.shape
        K = self.num_body_points
        device = cls_logits.device
        sigmas = self.sigmas.to(device=device, dtype=cls_logits.dtype)

        priors = torch.cat([anchor_pts, strides[:, None], strides[:, None]], dim=-1)  # [N, 4]

        total_cls      = cls_logits.new_zeros(())
        total_bbox     = cls_logits.new_zeros(())
        total_kpts     = cls_logits.new_zeros(())
        total_vis      = cls_logits.new_zeros(())
        total_bbox_aux = cls_logits.new_zeros(())
        num_fg = 0

        for b_idx in range(B):
            tgt = targets[b_idx]
            gt_boxes  = tgt["boxes"].to(device)
            gt_labels = tgt["labels"].to(device).long()
            gt_kpts   = tgt["keypoints"].to(device)
            gt_area   = tgt.get("area", None)

            img_size  = tgt["size"].to(device)
            H_img, W_img = img_size[0], img_size[1]
            scale_box  = torch.stack((W_img, H_img, W_img, H_img)).to(dtype=gt_boxes.dtype)
            gt_boxes   = box_cxcywh_to_xyxy(gt_boxes * scale_box)

            scale_kpts = torch.stack((W_img, H_img)).to(dtype=gt_kpts.dtype)
            gt_kpts_xy  = gt_kpts[:, :K * 2].view(-1, K, 2) * scale_kpts
            gt_kpts_vis_raw = gt_kpts[:, K * 2:]
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
                pred_keypoints=kpts_xy[b_idx].detach(),
                pred_keypoints_visible=kpts_vis[b_idx].detach(),
                gt_areas=gt_area,
                sigmas=sigmas,
                topk_candidates=self.topk,
                center_radius=self.center_radius,
                lambda_cls=self.assign_cls_weight,
                lambda_iou=self.assign_iou_weight,
                lambda_oks=self.assign_oks_weight,
                lambda_vis=0.0,
                dynamic_k_indicator=self.dynamic_k_indicator,
                use_keypoints_for_center=self.use_keypoints_for_center,
            )

            num_fg += assigned_gt.numel()

            target_score = torch.zeros(N, C, device=device, dtype=cls_logits.dtype)
            if assigned_gt.numel() > 0:
                cls_target = assigned_score.to(target_score).pow(self.overlaps_power).clamp(min=1e-3)
                target_score[fg_mask, assigned_cls] = cls_target

            total_cls = total_cls + varifocal_loss(cls_logits[b_idx], target_score)
            if assigned_gt.numel() == 0:
                continue

            # ── Foreground tensors ────────────────────────────────────────────
            pred_box_fg   = bbox_decoded[b_idx][fg_mask]   # [M, 4]
            pred_kpts_fg  = kpts_xy[b_idx][fg_mask]        # [M, K, 2]
            pred_vis_fg   = kpt_vis_raw[b_idx][fg_mask]    # [M, K]

            gt_kpts_fg_xy  = gt_kpts_xy[assigned_gt]
            gt_kpts_fg_vis = gt_kpts_visible[assigned_gt]
            gt_area_fg     = gt_area[assigned_gt]

            # ── Bbox IoU² loss ────────────────────────────────────────────────
            total_bbox = total_bbox + iou_loss_square(pred_box_fg, assigned_box).sum()

            # ── Auxiliary LTRB L1 loss ────────────────────────────────────────
            if self.use_aux_loss:
                pos_priors = priors[fg_mask]
                cx = pos_priors[:, 0]
                cy = pos_priors[:, 1]
                s  = pos_priors[:, 2].clamp(min=EPS)
                # Target normalized LTRB: (cx - x1)/s, (cy - y1)/s, (x2 - cx)/s, (y2 - cy)/s
                ltrb_target = torch.stack([
                    (cx - assigned_box[:, 0]) / s,
                    (cy - assigned_box[:, 1]) / s,
                    (assigned_box[:, 2] - cx) / s,
                    (assigned_box[:, 3] - cy) / s,
                ], dim=-1).clamp(min=0.0)
                # decode_ltrb uses softplus(raw) * stride, so softplus(raw)
                # is already the stride-normalized LTRB distance.
                ltrb_pred_norm = F.softplus(ltrb_raw[b_idx][fg_mask])
                total_bbox_aux = total_bbox_aux + F.l1_loss(
                    ltrb_pred_norm, ltrb_target, reduction="sum"
                )

            # ── OKS keypoint loss ─────────────────────────────────────────────
            total_kpts = total_kpts + oks_loss_rtmo(
                pred_kpts_fg, gt_kpts_fg_xy, gt_kpts_fg_vis, gt_area_fg, sigmas
            ).sum()

            # ── Visibility BCE ────────────────────────────────────────────────
            total_vis = total_vis + F.binary_cross_entropy_with_logits(
                pred_vis_fg, gt_kpts_fg_vis, reduction="sum"
            ) / max(K, 1)

        normaliser = max(num_fg, 1)
        loss_dict = {
            "loss_cls":  total_cls  / normaliser * self.weight_dict.get("loss_cls",  1.0),
            "loss_bbox": total_bbox / normaliser * self.weight_dict.get("loss_bbox", 5.0),
            "loss_kpts": total_kpts / normaliser * self.weight_dict.get("loss_kpts", 30.0),
            "loss_vis":  total_vis  / normaliser * self.weight_dict.get("loss_vis",  1.0),
        }
        if self.use_aux_loss:
            loss_dict["loss_bbox_aux"] = (
                total_bbox_aux / normaliser * self.weight_dict.get("loss_bbox_aux", 1.0)
            )
        return loss_dict
