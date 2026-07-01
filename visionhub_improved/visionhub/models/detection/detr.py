"""Detection-only matcher and postprocess for DETR-style models."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from torch import nn
from torchvision.ops import nms

from visionhub.misc.box_ops import box_cxcywh_to_xyxy, generalized_box_iou

try:
    import lapjv
except ModuleNotFoundError:
    lapjv = None
    try:
        from scipy.optimize import linear_sum_assignment as scipy_linear_sum_assignment
    except ModuleNotFoundError:
        scipy_linear_sum_assignment = None
else:
    scipy_linear_sum_assignment = None


_MATCHER_POOL = ThreadPoolExecutor(max_workers=min(os.cpu_count() or 4, 16))


def _lapjv_solve(cost_np):
    n_rows, n_cols = cost_np.shape
    if n_rows == 0 or n_cols == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    if lapjv is None:
        if scipy_linear_sum_assignment is None:
            raise ModuleNotFoundError(
                "HungarianBBoxMatcher requires either `lapjv` or `scipy`."
            )
        row_ind, col_ind = scipy_linear_sum_assignment(cost_np.astype(np.float64))
        return row_ind.astype(np.int64), col_ind.astype(np.int64)

    size = max(n_rows, n_cols)
    if n_rows != n_cols:
        padded = np.full((size, size), 1e9, dtype=np.float64)
        padded[:n_rows, :n_cols] = cost_np.astype(np.float64)
    else:
        padded = cost_np.astype(np.float64)
    _, x, _ = lapjv.lapjv(padded)
    row_ind = np.where(x[:n_rows] < n_cols)[0]
    col_ind = x[row_ind]
    return row_ind, col_ind


class HungarianBBoxMatcher(nn.Module):
    def __init__(self, cost_class=2.0, cost_bbox=5.0, cost_giou=2.0, focal_alpha=0.25):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.focal_alpha = focal_alpha

    @torch.no_grad()
    def forward(self, outputs, targets):
        bs, num_queries = outputs["pred_logits"].shape[:2]
        out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()
        out_boxes = outputs["pred_boxes"].flatten(0, 1)

        tgt_ids = torch.cat([v["labels"] for v in targets])
        sizes = [len(v["boxes"]) for v in targets]
        if tgt_ids.numel() == 0:
            return [
                (
                    torch.empty(0, dtype=torch.int64),
                    torch.empty(0, dtype=torch.int64),
                )
                for _ in targets
            ]

        tgt_boxes = torch.cat([v["boxes"] for v in targets])

        alpha = self.focal_alpha
        gamma = 2.0
        neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
        pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
        cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

        cost_bbox = torch.cdist(out_boxes, tgt_boxes, p=1)
        cost_giou = -generalized_box_iou(
            box_cxcywh_to_xyxy(out_boxes),
            box_cxcywh_to_xyxy(tgt_boxes),
        )

        cost = (
            self.cost_class * cost_class
            + self.cost_bbox * cost_bbox
            + self.cost_giou * cost_giou
        )
        cost = cost.view(bs, num_queries, -1).detach().cpu()

        cost_splits = cost.split(sizes, -1)
        futures = [
            _MATCHER_POOL.submit(_lapjv_solve, cost_splits[i][i].numpy())
            for i in range(len(sizes))
        ]
        indices = [f.result() for f in futures]
        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]


class DETRDetectionPostProcess(nn.Module):
    def __init__(self, num_select=300, nms_iou_threshold=0.65, use_nms=True) -> None:
        super().__init__()
        self.num_select = num_select
        self.nms_iou_threshold = nms_iou_threshold
        self.use_nms = use_nms
        self.deploy_mode = False

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        out_logits = outputs["pred_logits"]
        out_boxes = outputs["pred_boxes"]
        batch_size, _, num_classes = out_logits.shape
        num_select = min(self.num_select, out_logits.shape[1] * num_classes)

        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(batch_size, -1), num_select, dim=1)
        scores = topk_values
        topk_queries = torch.div(topk_indexes, num_classes, rounding_mode="floor")
        labels = topk_indexes % num_classes

        boxes = torch.gather(out_boxes, 1, topk_queries.unsqueeze(-1).expand(batch_size, num_select, 4))
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)[:, None, :]
        boxes = box_cxcywh_to_xyxy(boxes) * scale_fct

        if self.deploy_mode:
            return scores, labels, boxes

        results = []
        for s, l, b in zip(scores, labels, boxes):
            if self.use_nms and b.shape[0] > 0:
                max_coord = b.max()
                offsets = l.float() * (max_coord + 1)
                keep = nms(b + offsets[:, None], s, self.nms_iou_threshold)
                s, l, b = s[keep], l[keep], b[keep]
            results.append({"scores": s, "labels": l, "boxes": b})
        return results

    def deploy(self):
        self.eval()
        self.deploy_mode = True
        return self
