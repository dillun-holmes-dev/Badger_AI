"""Detection-only loss and postprocess helpers for dense families."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import batched_nms

from visionhub.misc.box_ops import box_cxcywh_to_xyxy
from visionhub.models.rtmo.assigner import simota_assign
from visionhub.models.rtmo.criterion import EPS, iou_loss_square, varifocal_loss


class DenseDetectionPostProcess(nn.Module):
    def __init__(
        self,
        score_threshold=0.3,
        nms_threshold=0.65,
        max_detections=300,
        pre_nms_topk=1000,
        min_box_size=2.0,
        min_box_area=4.0,
        class_score_margin=0.0,
        class_score_ratio=1.0,
        deploy_mode=False,
    ):
        super().__init__()
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.max_detections = max_detections
        self.pre_nms_topk = pre_nms_topk
        self.min_box_size = float(min_box_size)
        self.min_box_area = float(min_box_area)
        self.class_score_margin = float(class_score_margin)
        self.class_score_ratio = float(class_score_ratio)
        self.deploy_mode = deploy_mode

    def _select_class_scores(self, scores_all):
        scores, labels = scores_all.max(dim=-1)
        clear = torch.ones_like(scores, dtype=torch.bool)
        if scores_all.shape[-1] > 1 and (self.class_score_margin > 0 or self.class_score_ratio > 1):
            top2 = scores_all.topk(2, dim=-1).values
            runner_up = top2[..., 1]
            clear = (scores - runner_up >= self.class_score_margin) & (
                scores >= runner_up * self.class_score_ratio
            )
        return scores, labels, clear

    def _valid_box_mask(self, boxes):
        widths = boxes[..., 2] - boxes[..., 0]
        heights = boxes[..., 3] - boxes[..., 1]
        return (
            torch.isfinite(boxes).all(dim=-1)
            & (widths >= self.min_box_size)
            & (heights >= self.min_box_size)
            & (widths * heights >= self.min_box_area)
        )

    def _forward_deploy(self, outputs, orig_target_sizes):
        cls_logits = outputs["cls_logits"]
        bbox_decoded = outputs["bbox_decoded"]
        image_size = outputs["image_size"]

        scores_all = cls_logits.sigmoid()
        scores, labels, clear_class = self._select_class_scores(scores_all)
        batch_size, num_anchors = scores.shape
        topk = min(self.max_detections, self.pre_nms_topk)
        if not torch.onnx.is_in_onnx_export():
            topk = min(topk, num_anchors)

        topk_scores, topk_idx = scores.topk(topk, dim=1)
        topk_labels = torch.gather(labels, 1, topk_idx)
        topk_clear = torch.gather(clear_class, 1, topk_idx)
        topk_boxes = torch.gather(
            bbox_decoded,
            1,
            topk_idx.unsqueeze(-1).expand(batch_size, topk, 4),
        )

        score_mask = (topk_scores >= self.score_threshold) & topk_clear & self._valid_box_mask(topk_boxes)
        topk_scores = topk_scores * score_mask.to(topk_scores.dtype)
        topk_labels = topk_labels * score_mask.to(topk_labels.dtype)
        topk_boxes = topk_boxes * score_mask.unsqueeze(-1).to(topk_boxes.dtype)

        net_h = image_size[:, 0].float().unsqueeze(-1)
        net_w = image_size[:, 1].float().unsqueeze(-1)
        out_h = orig_target_sizes[:, 0].float().unsqueeze(-1)
        out_w = orig_target_sizes[:, 1].float().unsqueeze(-1)
        scale_x = (out_w / net_w.clamp(min=1)).unsqueeze(-1)
        scale_y = (out_h / net_h.clamp(min=1)).unsqueeze(-1)

        topk_boxes[..., 0::2] *= scale_x
        topk_boxes[..., 1::2] *= scale_y
        topk_boxes[..., 0::2].clamp_(min=0)
        topk_boxes[..., 1::2].clamp_(min=0)
        topk_boxes[..., 0::2] = torch.minimum(topk_boxes[..., 0::2], out_w.unsqueeze(-1))
        topk_boxes[..., 1::2] = torch.minimum(topk_boxes[..., 1::2], out_h.unsqueeze(-1))
        return topk_scores, topk_labels, topk_boxes

    @torch.no_grad()
    def forward(self, outputs, orig_target_sizes):
        if isinstance(outputs, list):
            return outputs

        if self.deploy_mode:
            return self._forward_deploy(outputs, orig_target_sizes)

        cls_logits = outputs["cls_logits"]
        bbox_decoded = outputs["bbox_decoded"]
        image_size = outputs["image_size"]
        batch_size = cls_logits.shape[0]
        results = []

        for b in range(batch_size):
            scores_all = cls_logits[b].sigmoid()
            scores, labels, clear_class = self._select_class_scores(scores_all)

            h_in, w_in = image_size[b, 0], image_size[b, 1]
            boxes = bbox_decoded[b].clone()
            boxes[:, 0::2].clamp_(0, w_in)
            boxes[:, 1::2].clamp_(0, h_in)

            keep = (scores >= self.score_threshold) & clear_class & self._valid_box_mask(boxes)
            if not keep.any():
                results.append(
                    {
                        "scores": scores.new_zeros((0,)),
                        "labels": labels.new_zeros((0,)),
                        "boxes": boxes.new_zeros((0, 4)),
                    }
                )
                continue

            scores = scores[keep]
            labels = labels[keep]
            boxes = boxes[keep]

            if scores.shape[0] > self.pre_nms_topk:
                topk_idx = scores.topk(self.pre_nms_topk).indices
                scores = scores[topk_idx]
                labels = labels[topk_idx]
                boxes = boxes[topk_idx]

            keep_idx = batched_nms(boxes, scores, labels, self.nms_threshold)
            keep_idx = keep_idx[: self.max_detections]
            scores = scores[keep_idx]
            labels = labels[keep_idx]
            boxes = boxes[keep_idx]

            h_out, w_out = orig_target_sizes[b, 0].float(), orig_target_sizes[b, 1].float()
            scale_x = w_out / w_in.clamp(min=1)
            scale_y = h_out / h_in.clamp(min=1)
            boxes[:, 0::2] *= scale_x
            boxes[:, 1::2] *= scale_y
            boxes[:, 0::2].clamp_(0, w_out)
            boxes[:, 1::2].clamp_(0, h_out)

            results.append({"scores": scores, "labels": labels, "boxes": boxes})

        return results


class RTMODetectionCriterion(nn.Module):
    def __init__(
        self,
        num_classes,
        weight_dict=None,
        topk_candidates=10,
        center_radius=2.5,
        assign_cls_weight=1.0,
        assign_iou_weight=3.0,
        overlaps_power=0.5,
        use_aux_loss=True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.weight_dict = weight_dict or {
            "loss_cls": 2.0,
            "loss_bbox": 5.0,
            "loss_bbox_aux": 1.0,
        }
        self.topk = topk_candidates
        self.center_radius = center_radius
        self.assign_cls_weight = assign_cls_weight
        self.assign_iou_weight = assign_iou_weight
        self.overlaps_power = overlaps_power
        self.use_aux_loss = use_aux_loss

    def forward(self, outputs, targets):
        cls_logits = outputs["cls_logits"]
        bbox_pred = outputs["bbox_pred"]
        bbox_decoded = outputs["bbox_decoded"]
        anchor_pts = outputs["anchor_pts"]
        strides = outputs["strides"]

        batch_size, num_anchors, num_classes = cls_logits.shape
        device = cls_logits.device
        priors = torch.cat([anchor_pts, strides[:, None], strides[:, None]], dim=-1)

        total_cls = cls_logits.new_zeros(())
        total_bbox = cls_logits.new_zeros(())
        total_bbox_aux = cls_logits.new_zeros(())
        num_fg = 0

        for b_idx in range(batch_size):
            tgt = targets[b_idx]
            gt_boxes = tgt["boxes"].to(device)
            gt_labels = tgt["labels"].to(device).long()

            img_size = tgt["size"].to(device)
            h_img, w_img = img_size[0], img_size[1]
            scale_box = torch.stack((w_img, h_img, w_img, h_img)).to(dtype=gt_boxes.dtype)
            gt_boxes = box_cxcywh_to_xyxy(gt_boxes * scale_box)

            cls_scores = cls_logits[b_idx].detach().clamp(min=-1e4, max=1e4).sigmoid().sqrt()
            fg_mask, assigned_gt, assigned_cls, assigned_box, assigned_score = simota_assign(
                cls_scores,
                bbox_decoded[b_idx].detach(),
                priors,
                gt_boxes,
                gt_labels,
                gt_keypoints=None,
                gt_keypoints_visible=None,
                pred_keypoints=None,
                pred_keypoints_visible=None,
                gt_areas=None,
                sigmas=None,
                topk_candidates=self.topk,
                center_radius=self.center_radius,
                lambda_cls=self.assign_cls_weight,
                lambda_iou=self.assign_iou_weight,
                lambda_oks=0.0,
                lambda_vis=0.0,
                dynamic_k_indicator="iou",
                use_keypoints_for_center=False,
            )

            num_fg += assigned_gt.numel()
            target_score = torch.zeros(num_anchors, num_classes, device=device, dtype=cls_logits.dtype)
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

        normaliser = max(num_fg, 1)
        losses = {
            "loss_cls": total_cls / normaliser * self.weight_dict.get("loss_cls", 2.0),
            "loss_bbox": total_bbox / normaliser * self.weight_dict.get("loss_bbox", 5.0),
        }
        if self.use_aux_loss:
            losses["loss_bbox_aux"] = (
                total_bbox_aux / normaliser * self.weight_dict.get("loss_bbox_aux", 1.0)
            )
        return losses


class RTMDetDetectionCriterion(nn.Module):
    def __init__(
        self,
        num_classes,
        weight_dict=None,
        topk_candidates=10,
        center_radius=2.5,
        assign_cls_weight=2.0,
        assign_iou_weight=3.0,
        overlaps_power=0.5,
        use_aux_loss=True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.weight_dict = weight_dict or {
            "loss_cls": 2.0,
            "loss_bbox": 5.0,
            "loss_bbox_aux": 1.0,
        }
        self.topk = topk_candidates
        self.center_radius = center_radius
        self.assign_cls_weight = assign_cls_weight
        self.assign_iou_weight = assign_iou_weight
        self.overlaps_power = overlaps_power
        self.use_aux_loss = use_aux_loss

    def forward(self, outputs, targets):
        cls_logits = outputs["cls_logits"]
        ltrb_raw = outputs["bbox_pred"]
        bbox_decoded = outputs["bbox_decoded"]
        anchor_pts = outputs["anchor_pts"]
        strides = outputs["strides"]

        batch_size, num_anchors, num_classes = cls_logits.shape
        device = cls_logits.device
        priors = torch.cat([anchor_pts, strides[:, None], strides[:, None]], dim=-1)

        total_cls = cls_logits.new_zeros(())
        total_bbox = cls_logits.new_zeros(())
        total_bbox_aux = cls_logits.new_zeros(())
        num_fg = 0

        for b_idx in range(batch_size):
            tgt = targets[b_idx]
            gt_boxes = tgt["boxes"].to(device)
            gt_labels = tgt["labels"].to(device).long()

            img_size = tgt["size"].to(device)
            h_img, w_img = img_size[0], img_size[1]
            scale_box = torch.stack((w_img, h_img, w_img, h_img)).to(dtype=gt_boxes.dtype)
            gt_boxes = box_cxcywh_to_xyxy(gt_boxes * scale_box)

            cls_scores = cls_logits[b_idx].detach().clamp(min=-1e4, max=1e4).sigmoid().sqrt()
            fg_mask, assigned_gt, assigned_cls, assigned_box, assigned_score = simota_assign(
                cls_scores,
                bbox_decoded[b_idx].detach(),
                priors,
                gt_boxes,
                gt_labels,
                gt_keypoints=None,
                gt_keypoints_visible=None,
                pred_keypoints=None,
                pred_keypoints_visible=None,
                gt_areas=None,
                sigmas=None,
                topk_candidates=self.topk,
                center_radius=self.center_radius,
                lambda_cls=self.assign_cls_weight,
                lambda_iou=self.assign_iou_weight,
                lambda_oks=0.0,
                lambda_vis=0.0,
                dynamic_k_indicator="iou",
                use_keypoints_for_center=False,
            )

            num_fg += assigned_gt.numel()
            target_score = torch.zeros(num_anchors, num_classes, device=device, dtype=cls_logits.dtype)
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
                cx = pos_priors[:, 0]
                cy = pos_priors[:, 1]
                s = pos_priors[:, 2].clamp(min=EPS)
                ltrb_target = torch.stack(
                    [
                        (cx - assigned_box[:, 0]) / s,
                        (cy - assigned_box[:, 1]) / s,
                        (assigned_box[:, 2] - cx) / s,
                        (assigned_box[:, 3] - cy) / s,
                    ],
                    dim=-1,
                ).clamp(min=0.0)
                ltrb_pred_norm = F.softplus(ltrb_raw[b_idx][fg_mask])
                total_bbox_aux = total_bbox_aux + F.l1_loss(
                    ltrb_pred_norm, ltrb_target, reduction="sum"
                )

        normaliser = max(num_fg, 1)
        losses = {
            "loss_cls": total_cls / normaliser * self.weight_dict.get("loss_cls", 2.0),
            "loss_bbox": total_bbox / normaliser * self.weight_dict.get("loss_bbox", 5.0),
        }
        if self.use_aux_loss:
            losses["loss_bbox_aux"] = (
                total_bbox_aux / normaliser * self.weight_dict.get("loss_bbox_aux", 1.0)
            )
        return losses
