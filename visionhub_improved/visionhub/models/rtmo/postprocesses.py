"""
RTMOPose PostProcessor.
Converts raw model outputs → same format as DETRPose PostProcess:
  [{'scores': [N], 'labels': [N], 'boxes': [N, 4], 'keypoints': [N, K*3]}]
Boxes are x1y1x2y2 absolute pixels.
Keypoints are [x1, y1, vis1, x2, y2, vis2, ...] absolute pixels.
"""

import torch
import torch.nn as nn
from torchvision.ops import batched_nms, box_iou

from .criterion import bbox_xyxy_to_cs


class RTMOPostProcess(nn.Module):
    """
    Args:
        score_threshold:  Pre-NMS confidence filter.
        nms_threshold:    IoU threshold for NMS.
        max_detections:   Keep at most this many detections per image.
        deploy_mode:      If True return raw tensors (for TensorRT export).
    """

    def __init__(
        self,
        score_threshold=0.3,
        nms_threshold=0.65,
        max_detections=300,
        pre_nms_topk=1000,
        bbox_padding=1.25,
        use_pose_nms=True,
        pose_keypoint_threshold=0.25,
        pose_center_threshold=0.75,
        keypoint_vis_threshold=0.3,
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
        self.bbox_padding = bbox_padding
        self.use_pose_nms = use_pose_nms
        self.pose_keypoint_threshold = pose_keypoint_threshold
        self.pose_center_threshold = pose_center_threshold
        self.keypoint_vis_threshold = keypoint_vis_threshold
        self.min_box_size = float(min_box_size)
        self.min_box_area = float(min_box_area)
        self.class_score_margin = float(class_score_margin)
        self.class_score_ratio = float(class_score_ratio)
        self.deploy_mode = deploy_mode
        object.__setattr__(self, "_dcc_ref", None)

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

    @property
    def dcc(self):
        return self._dcc_ref

    def set_dcc(self, dcc):
        object.__setattr__(self, "_dcc_ref", dcc)
        return self

    @staticmethod
    def _bbox_xyxy_to_cs_batched(boxes_xyxy, padding):
        center = (boxes_xyxy[..., :2] + boxes_xyxy[..., 2:]) * 0.5
        scale = (boxes_xyxy[..., 2:] - boxes_xyxy[..., :2]).clamp(min=1.0) * padding
        return torch.cat([center, scale], dim=-1)

    def _pose_duplicate_suppression(self, boxes, kpts, vis, labels):
        """Remove same-class pose duplicates that box IoU NMS misses."""
        num_dets = boxes.shape[0]
        if num_dets <= 1 or not self.use_pose_nms or self.deploy_mode:
            return torch.arange(num_dets, device=boxes.device)

        calc_boxes = boxes.float()
        calc_kpts = kpts.float()
        calc_vis = vis.float()
        keep = torch.ones(num_dets, dtype=torch.bool, device=boxes.device)
        centers = (calc_boxes[:, :2] + calc_boxes[:, 2:]) * 0.5
        wh = (calc_boxes[:, 2:] - calc_boxes[:, :2]).clamp(min=1.0)
        scale = wh.prod(dim=-1).sqrt().clamp(min=1.0)
        ious = box_iou(calc_boxes, calc_boxes)

        for i in range(num_dets):
            if not keep[i]:
                continue

            rest = torch.arange(i + 1, num_dets, device=boxes.device)
            rest = rest[keep[i + 1:]]
            if rest.numel() == 0:
                continue

            same_label = labels[rest] == labels[i]
            suppress = same_label & (ious[i, rest] > self.nms_threshold)

            visible = (calc_vis[i] > self.keypoint_vis_threshold) & (
                calc_vis[rest] > self.keypoint_vis_threshold
            )
            visible_count = visible.sum(dim=-1)
            has_visible_overlap = visible_count > 0

            kpt_dist = torch.linalg.vector_norm(calc_kpts[rest] - calc_kpts[i], dim=-1)
            mean_kpt_dist = (kpt_dist * visible.to(kpt_dist.dtype)).sum(dim=-1)
            mean_kpt_dist = mean_kpt_dist / visible_count.clamp(min=1).to(kpt_dist.dtype)
            norm_kpt_dist = mean_kpt_dist / torch.minimum(scale[i], scale[rest])

            center_dist = torch.linalg.vector_norm(centers[rest] - centers[i], dim=-1)
            norm_center_dist = center_dist / torch.minimum(scale[i], scale[rest])

            pose_duplicate = (
                same_label
                & has_visible_overlap
                & (norm_kpt_dist < self.pose_keypoint_threshold)
                & (norm_center_dist < self.pose_center_threshold)
            )
            suppress = suppress | pose_duplicate
            keep[rest[suppress]] = False

        return torch.nonzero(keep, as_tuple=False).flatten()

    def _forward_deploy(self, outputs, orig_target_sizes):
        """Return fixed-shape tensors for ONNX/TensorRT export."""
        cls_logits = outputs["cls_logits"]      # [B, N, C]
        bbox_decoded = outputs["bbox_decoded"]  # [B, N, 4]
        kpts_decoded = outputs["kpts_decoded"]  # [B, N, K, 2]
        kpts_vis = outputs["kpts_vis"]          # [B, N, K]
        image_size = outputs["image_size"]      # [B, 2] H,W net-input

        scores_all = cls_logits.sigmoid()
        scores, labels, clear_class = self._select_class_scores(scores_all)

        batch_size, num_anchors = scores.shape
        num_body_points = kpts_decoded.shape[2]
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
        topk_kpts = torch.gather(
            kpts_decoded,
            1,
            topk_idx.unsqueeze(-1).unsqueeze(-1).expand(batch_size, topk, num_body_points, 2),
        )
        topk_vis = torch.gather(
            kpts_vis,
            1,
            topk_idx.unsqueeze(-1).expand(batch_size, topk, num_body_points),
        )

        pose_vecs = outputs.get("pose_vecs", None)
        anchor_pts = outputs.get("anchor_pts", None)
        if self.dcc is not None and pose_vecs is not None and anchor_pts is not None:
            topk_pose_vecs = torch.gather(
                pose_vecs,
                1,
                topk_idx.unsqueeze(-1).expand(batch_size, topk, pose_vecs.shape[-1]),
            )
            anchor_pts_batched = anchor_pts.unsqueeze(0).expand(batch_size, num_anchors, 2)
            topk_grids = torch.gather(
                anchor_pts_batched,
                1,
                topk_idx.unsqueeze(-1).expand(batch_size, topk, 2),
            )
            bbox_cs = self._bbox_xyxy_to_cs_batched(topk_boxes, self.bbox_padding)
            topk_kpts = self.dcc.forward_test(topk_pose_vecs, bbox_cs, topk_grids)

        score_mask = (topk_scores >= self.score_threshold) & topk_clear & self._valid_box_mask(topk_boxes)
        topk_scores = topk_scores * score_mask.to(topk_scores.dtype)
        topk_labels = topk_labels * score_mask.to(topk_labels.dtype)
        topk_boxes = topk_boxes * score_mask.unsqueeze(-1).to(topk_boxes.dtype)
        topk_kpts = topk_kpts * score_mask.unsqueeze(-1).unsqueeze(-1).to(topk_kpts.dtype)
        topk_vis = topk_vis * score_mask.unsqueeze(-1).to(topk_vis.dtype)

        net_h = image_size[:, 0].float().unsqueeze(-1)
        net_w = image_size[:, 1].float().unsqueeze(-1)
        out_h = orig_target_sizes[:, 0].float().unsqueeze(-1)
        out_w = orig_target_sizes[:, 1].float().unsqueeze(-1)

        scale_x = (out_w / net_w.clamp(min=1)).unsqueeze(-1)
        scale_y = (out_h / net_h.clamp(min=1)).unsqueeze(-1)

        topk_boxes[..., 0::2] *= scale_x
        topk_boxes[..., 1::2] *= scale_y
        topk_kpts[..., 0] *= scale_x
        topk_kpts[..., 1] *= scale_y

        topk_boxes[..., 0::2].clamp_(min=0)
        topk_boxes[..., 1::2].clamp_(min=0)
        topk_boxes[..., 0::2] = torch.minimum(topk_boxes[..., 0::2], out_w.unsqueeze(-1))
        topk_boxes[..., 1::2] = torch.minimum(topk_boxes[..., 1::2], out_h.unsqueeze(-1))
        topk_kpts[..., 0] = torch.minimum(torch.clamp(topk_kpts[..., 0], min=0), out_w.unsqueeze(-1))
        topk_kpts[..., 1] = torch.minimum(torch.clamp(topk_kpts[..., 1], min=0), out_h.unsqueeze(-1))

        keypoints = torch.cat(
            [topk_kpts[..., 0:1], topk_kpts[..., 1:2], topk_vis.unsqueeze(-1)],
            dim=-1,
        ).reshape(batch_size, topk, num_body_points * 3)

        return topk_scores, topk_labels, keypoints, topk_boxes

    @torch.no_grad()
    def forward(self, outputs, orig_target_sizes):
        """
        Args:
            outputs: dict from RTMOPose.forward() (training mode) with keys:
                'cls_logits'   [B, N, C]
                'bbox_decoded' [B, N, 4]  xyxy absolute (in net input space)
                'kpts_decoded' [B, N, K, 2] proxy absolute
                'kpts_vis'     [B, N, K]  vis scores (sigmoid)
                'pose_vecs'     [B, N, C]  optional DCC pose vectors
                'anchor_pts'   [N, 2]
                'strides'      [N]
                'image_size'   [B, 2] H,W input tensor size
                OR a list of result dicts (when called after model.eval() inline).
            orig_target_sizes: [B, 2] tensor of (H, W) for each image
                               (used to clamp / scale back if the model
                                was given a resized image — pass the net
                                input size if you don't want scaling).
        Returns:
            list of dicts, one per image.
        """
        # When model is in eval mode it returns results list directly;
        # this postprocessor may be called as a passthrough from the engine.
        if isinstance(outputs, list):
            return outputs

        if self.deploy_mode:
            return self._forward_deploy(outputs, orig_target_sizes)

        cls_logits   = outputs["cls_logits"]    # [B, N, C]
        bbox_decoded = outputs["bbox_decoded"]   # [B, N, 4]
        kpts_decoded = outputs["kpts_decoded"]   # [B, N, K, 2]
        kpts_vis     = outputs["kpts_vis"]       # [B, N, K]
        input_sizes   = outputs.get("image_size", None)

        B = cls_logits.shape[0]
        scores_all = cls_logits.sigmoid()  # [B, N, C]

        results = []
        for b in range(B):
            scores = scores_all[b]    # [N, C]
            boxes  = bbox_decoded[b]  # [N, 4]
            kpts   = kpts_decoded[b]  # [N, K, 2]
            vis    = kpts_vis[b]      # [N, K]
            pose_vecs = outputs.get("pose_vecs", None)
            grids = outputs.get("anchor_pts", None)

            # Best score and class per anchor
            max_scores, labels, clear_class = self._select_class_scores(scores)  # [N], [N]
            boxes = boxes.clone()
            if input_sizes is not None:
                in_h, in_w = input_sizes[b, 0], input_sizes[b, 1]
                boxes[:, 0::2].clamp_(0, in_w)
                boxes[:, 1::2].clamp_(0, in_h)

            # Pre-NMS score threshold
            keep = (max_scores >= self.score_threshold) & clear_class & self._valid_box_mask(boxes)
            if keep.sum() == 0:
                results.append({
                    "scores":    torch.zeros(0, device=cls_logits.device),
                    "labels":    torch.zeros(0, dtype=torch.long, device=cls_logits.device),
                    "boxes":     torch.zeros(0, 4, device=cls_logits.device),
                    "keypoints": torch.zeros(0, kpts.shape[1] * 3, device=cls_logits.device),
                })
                continue

            max_scores = max_scores[keep]
            labels     = labels[keep]
            boxes      = boxes[keep]
            kpts       = kpts[keep]
            vis        = vis[keep]
            if pose_vecs is not None:
                pose_vecs_b = pose_vecs[b][keep]
            else:
                pose_vecs_b = None
            if grids is not None:
                grids_b = grids[keep]
            else:
                grids_b = None

            # Pre-NMS top-K: cap candidates to avoid slow NMS when many boxes pass threshold
            if max_scores.shape[0] > self.pre_nms_topk:
                _, topk_idx = max_scores.topk(self.pre_nms_topk)
                max_scores = max_scores[topk_idx]
                labels     = labels[topk_idx]
                boxes      = boxes[topk_idx]
                kpts       = kpts[topk_idx]
                vis        = vis[topk_idx]
                if pose_vecs_b is not None:
                    pose_vecs_b = pose_vecs_b[topk_idx]
                if grids_b is not None:
                    grids_b = grids_b[topk_idx]

            # NMS (class-aware via offset)
            keep_nms = batched_nms(boxes, max_scores, labels, self.nms_threshold)
            if keep_nms.shape[0] > self.max_detections:
                keep_nms = keep_nms[:self.max_detections]

            max_scores = max_scores[keep_nms]
            labels     = labels[keep_nms]
            boxes      = boxes[keep_nms]
            kpts       = kpts[keep_nms]      # [M, K, 2]
            vis        = vis[keep_nms]       # [M, K]
            if pose_vecs_b is not None:
                pose_vecs_b = pose_vecs_b[keep_nms]
            if grids_b is not None:
                grids_b = grids_b[keep_nms]

            # MMPose RTMO decodes final keypoints with DCC after score filtering
            # and NMS. If no DCC has been linked, the proxy offsets remain the
            # fallback so old inference/export paths still work.
            if self.dcc is not None and pose_vecs_b is not None and grids_b is not None and boxes.numel() > 0:
                bbox_cs = bbox_xyxy_to_cs(boxes, padding=self.bbox_padding)
                kpts = self.dcc.forward_test(pose_vecs_b, bbox_cs, grids_b)

            pose_keep = self._pose_duplicate_suppression(boxes, kpts, vis, labels)
            max_scores = max_scores[pose_keep]
            labels     = labels[pose_keep]
            boxes      = boxes[pose_keep]
            kpts       = kpts[pose_keep]
            vis        = vis[pose_keep]

            # Scale from network input space to requested output image space.
            if orig_target_sizes is not None:
                h, w = orig_target_sizes[b, 0], orig_target_sizes[b, 1]
                if input_sizes is not None:
                    in_h, in_w = input_sizes[b, 0].to(h.device), input_sizes[b, 1].to(w.device)
                else:
                    stride = outputs["strides"].max()
                    anchors = outputs["anchor_pts"]
                    in_w = anchors[:, 0].max() + stride / 2
                    in_h = anchors[:, 1].max() + stride / 2
                scale_x = w / in_w.clamp(min=1)
                scale_y = h / in_h.clamp(min=1)
                boxes[:, 0::2] = boxes[:, 0::2] * scale_x
                boxes[:, 1::2] = boxes[:, 1::2] * scale_y
                kpts[..., 0] = kpts[..., 0] * scale_x
                kpts[..., 1] = kpts[..., 1] * scale_y
                h, w = h.item(), w.item()
                boxes[:, 0::2] = boxes[:, 0::2].clamp(0, w)
                boxes[:, 1::2] = boxes[:, 1::2].clamp(0, h)
                kpts[..., 0] = kpts[..., 0].clamp(0, w)
                kpts[..., 1] = kpts[..., 1].clamp(0, h)

            # Build keypoints tensor: [M, K*3] → [x, y, vis, x, y, vis, ...]
            M, K, _ = kpts.shape
            kpts_flat = torch.cat([kpts, vis.unsqueeze(-1)], dim=-1)  # [M, K, 3]
            kpts_flat = kpts_flat.reshape(M, K * 3)

            if self.deploy_mode:
                results.append({
                    "scores":    max_scores,
                    "labels":    labels,
                    "boxes":     boxes,
                    "keypoints": kpts_flat,
                })
            else:
                results.append({
                    "scores":    max_scores,
                    "labels":    labels,
                    "boxes":     boxes,
                    "keypoints": kpts_flat,
                })

        return results
