"""
RTMDet-Pose PostProcessor.
Converts raw model outputs → list of dicts (same format as RTMO / DETRPose):
  [{'scores': [N], 'labels': [N], 'boxes': [N, 4], 'keypoints': [N, K*3]}]
Boxes are x1y1x2y2 absolute pixels.
Keypoints are [x1, y1, vis1, x2, y2, vis2, ...] absolute pixels.
"""

import torch
import torch.nn as nn
from torchvision.ops import batched_nms, box_iou


class RTMDetPostProcess(nn.Module):
    """
    Args:
        score_threshold:  Pre-NMS confidence filter.
        nms_threshold:    IoU threshold for NMS.
        max_detections:   Max detections per image.
        pre_nms_topk:     Keep top-k before NMS.
        deploy_mode:      If True return raw tensors (for TensorRT export).
    """

    def __init__(
        self,
        score_threshold: float = 0.3,
        nms_threshold: float = 0.65,
        max_detections: int = 300,
        pre_nms_topk: int = 1000,
        use_pose_nms: bool = True,
        pose_keypoint_threshold: float = 0.25,
        pose_center_threshold: float = 0.75,
        keypoint_vis_threshold: float = 0.3,
        min_box_size: float = 2.0,
        min_box_area: float = 4.0,
        class_score_margin: float = 0.0,
        class_score_ratio: float = 1.0,
        deploy_mode: bool = False,
    ):
        super().__init__()
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.max_detections = max_detections
        self.pre_nms_topk = pre_nms_topk
        self.use_pose_nms = use_pose_nms
        self.pose_keypoint_threshold = pose_keypoint_threshold
        self.pose_center_threshold = pose_center_threshold
        self.keypoint_vis_threshold = keypoint_vis_threshold
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

    def _forward_deploy(self, cls_logits, bbox_decoded, kpts_decoded, kpts_vis, image_size, orig_target_sizes):
        """Return fixed-shape tensors for ONNX/TensorRT export."""
        scores_all = cls_logits.sigmoid()      # [B, N, C]
        scores, labels, clear_class = self._select_class_scores(scores_all)  # [B, N], [B, N]

        batch_size, num_anchors = scores.shape
        num_body_points = kpts_decoded.shape[2]
        # Keep the exported graph free of Python comparisons on symbolic shape
        # values. For the fixed 640x640 deployment path, anchors comfortably
        # exceed this limit, so the export-time top-k can stay constant.
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
            outputs: dict from RTMDetPose.forward() or list of result dicts.
            orig_target_sizes: [B, 2] (H, W) for each image.
        Returns:
            list of dicts, one per image, with keys:
              'scores'    [M]
              'labels'    [M]
              'boxes'     [M, 4]  xyxy
              'keypoints' [M, K*3]  [x1,y1,v1, x2,y2,v2, ...]
        """
        if isinstance(outputs, list):
            return outputs

        cls_logits   = outputs["cls_logits"]    # [B, N, C]
        bbox_decoded = outputs["bbox_decoded"]   # [B, N, 4]  xyxy
        kpts_decoded = outputs["kpts_decoded"]   # [B, N, K, 2]
        kpts_vis     = outputs["kpts_vis"]       # [B, N, K]
        image_size   = outputs["image_size"]     # [B, 2]  H, W net-input

        if self.deploy_mode:
            return self._forward_deploy(
                cls_logits,
                bbox_decoded,
                kpts_decoded,
                kpts_vis,
                image_size,
                orig_target_sizes,
            )

        B = cls_logits.shape[0]
        K = kpts_decoded.shape[2]
        results = []

        for b in range(B):
            scores_all = cls_logits[b].sigmoid()        # [N, C]
            scores, labels, clear_class = self._select_class_scores(scores_all)     # [N]

            # Clamp to input image bounds
            H_in, W_in = image_size[b, 0], image_size[b, 1]
            boxes = bbox_decoded[b].clone()
            boxes[:, 0::2].clamp_(0, W_in)
            boxes[:, 1::2].clamp_(0, H_in)

            # Score threshold
            keep = (scores >= self.score_threshold) & clear_class & self._valid_box_mask(boxes)
            if not keep.any():
                H_out, W_out = orig_target_sizes[b, 0], orig_target_sizes[b, 1]
                results.append({
                    "scores":     scores.new_zeros((0,)),
                    "labels":     labels.new_zeros((0,)),
                    "boxes":      boxes.new_zeros((0, 4)),
                    "keypoints":  boxes.new_zeros((0, K * 3)),
                })
                continue

            scores = scores[keep]
            labels = labels[keep]
            boxes  = boxes[keep]
            kpts   = kpts_decoded[b][keep]   # [M, K, 2]
            vis    = kpts_vis[b][keep]        # [M, K]

            # Top-k before NMS
            if scores.shape[0] > self.pre_nms_topk:
                topk_idx = scores.topk(self.pre_nms_topk).indices
                scores = scores[topk_idx]
                labels = labels[topk_idx]
                boxes  = boxes[topk_idx]
                kpts   = kpts[topk_idx]
                vis    = vis[topk_idx]

            # NMS
            keep_idx = batched_nms(boxes, scores, labels, self.nms_threshold)
            keep_idx = keep_idx[:self.max_detections]
            scores = scores[keep_idx]
            labels = labels[keep_idx]
            boxes  = boxes[keep_idx]
            kpts   = kpts[keep_idx]
            vis    = vis[keep_idx]

            pose_keep = self._pose_duplicate_suppression(boxes, kpts, vis, labels)
            scores = scores[pose_keep]
            labels = labels[pose_keep]
            boxes  = boxes[pose_keep]
            kpts   = kpts[pose_keep]
            vis    = vis[pose_keep]

            # Scale from network-input space to original image space
            H_in, W_in   = image_size[b, 0].float(), image_size[b, 1].float()
            H_out, W_out = orig_target_sizes[b, 0].float(), orig_target_sizes[b, 1].float()
            scale_x = W_out / W_in.clamp(min=1)
            scale_y = H_out / H_in.clamp(min=1)

            boxes[:, 0::2] *= scale_x
            boxes[:, 1::2] *= scale_y
            kpts[..., 0]   *= scale_x
            kpts[..., 1]   *= scale_y
            boxes[:, 0::2].clamp_(0, W_out)
            boxes[:, 1::2].clamp_(0, H_out)
            kpts[..., 0].clamp_(0, W_out)
            kpts[..., 1].clamp_(0, H_out)

            # Pack keypoints: [M, K*3]  [x,y,vis, x,y,vis, ...]
            keypoints = torch.cat(
                [kpts[..., 0:1], kpts[..., 1:2], vis.unsqueeze(-1)], dim=-1
            ).reshape(-1, K * 3)

            results.append({
                "scores":     scores,
                "labels":     labels,
                "boxes":      boxes,
                "keypoints":  keypoints,
            })

        return results
