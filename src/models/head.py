"""
Detection heads for Badger.

The head takes fused features from the neck and produces:
  1. Classification scores — "what object is this?"
  2. Bounding box coordinates — "where is it?"
"""

import torch
import torch.nn as nn
from .blocks import Conv, DFL


class DecoupledHead(nn.Module):
    """
    Decoupled detection head — Badger's default.

    "Decoupled" means classification and regression use separate convolutional
    branches. This is better than a shared head because:
      - Classification needs to know WHAT (semantic features)
      - Regression needs to know WHERE (spatial features)
    These tasks benefit from different feature representations.
    """
    def __init__(self, num_classes=80, channels=None, reg_max=16):
        super().__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.num_outputs = num_classes
        self.channels = channels or [256, 256, 256]  # Neck outputs same channel count

        self.cls_branches = nn.ModuleList()
        self.reg_branches = nn.ModuleList()

        for ch in self.channels:
            # Classification branch: two 3x3 convs + 1x1 output
            cls_branch = nn.Sequential(
                Conv(ch, ch, 3),
                Conv(ch, ch, 3),
                nn.Conv2d(ch, self.num_outputs, 1)
            )
            self.cls_branches.append(cls_branch)

            # Regression branch: two 3x3 convs + 1x1 output (4 * reg_max)
            # 4 = (left, top, right, bottom), reg_max = bins per edge
            reg_branch = nn.Sequential(
                Conv(ch, ch, 3),
                Conv(ch, ch, 3),
                nn.Conv2d(ch, 4 * reg_max, 1)
            )
            self.reg_branches.append(reg_branch)

        self.dfl = DFL(reg_max)

        self._init_biases()

    def _init_biases(self):
        """Initialize with small negative bias → models start conservative."""
        for cls_branch, reg_branch in zip(self.cls_branches, self.reg_branches):
            nn.init.constant_(cls_branch[-1].bias, -4.0)
            nn.init.constant_(reg_branch[-1].bias, 0.0)

    def forward(self, features):
        """
        Args:
            features: [N3, N4, N5] from neck
                      Shapes: [B, C, 80, 80], [B, C, 40, 40], [B, C, 20, 20]

        Returns:
            cls_scores: list of [B, num_classes, H, W]
            bbox_preds: list of [B, 4, H, W]
        """
        cls_scores = []
        bbox_preds = []

        for i, feat in enumerate(features):
            cls_out = self.cls_branches[i](feat)
            reg_out = self.reg_branches[i](feat)
            bbox_out = self.dfl(reg_out)
            cls_scores.append(cls_out)
            bbox_preds.append(bbox_out)

        return cls_scores, bbox_preds


class CoupledHead(nn.Module):
    """
    Coupled (shared) detection head — YOLOv5 style.

    Uses a single branch to predict both class and box.
    Simpler and faster, but generally less accurate than decoupled.

    Improvement experiment: compare decoupled vs. coupled to measure the
    accuracy/speed tradeoff for your specific use case.
    """
    def __init__(self, num_classes=80, channels=None):
        super().__init__()
        self.num_classes = num_classes
        self.num_outputs = num_classes + 5  # classes + (x, y, w, h, obj)
        self.channels = channels or [256, 256, 256]

        self.branches = nn.ModuleList()
        for ch in self.channels:
            branch = nn.Sequential(
                Conv(ch, ch, 3),
                nn.Conv2d(ch, self.num_outputs, 1)
            )
            self.branches.append(branch)

    def forward(self, features):
        outputs = []
        for i, feat in enumerate(features):
            outputs.append(self.branches[i](feat))
        return outputs


# =============================================================================
# NMS-Free Dual Head (YOLOv10/YOLO26)
# =============================================================================

from scipy.optimize import linear_sum_assignment


class DualHead(nn.Module):
    """
    Dual Detection Head — one2many (training) + one2one (inference).

    During training:
      - Both heads produce predictions
      - one2many head uses TAL/SimOTA (standard YOLO assignment)
      - one2one head uses Hungarian matching (DETR-style)
      - Total loss = L_one2many + λ × L_one2one

    During inference:
      - Only one2one head runs → no NMS needed
      - Each ground truth gets exactly one prediction

    The consistent matching strategy ensures both heads learn
    complementary representations:
      - one2many: learns general features (good for training)
      - one2one: learns precise, non-redundant features (good for inference)
    """

    def __init__(self, num_classes=80, channels=None, reg_max=16):
        super().__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.channels = channels or [256, 256, 256]

        # ---- one2many head (standard decoupled — used for training) ----
        self.m_cls_branches = nn.ModuleList()
        self.m_reg_branches = nn.ModuleList()
        for ch in self.channels:
            self.m_cls_branches.append(nn.Sequential(
                Conv(ch, ch, 3), Conv(ch, ch, 3),
                nn.Conv2d(ch, num_classes, 1)
            ))
            self.m_reg_branches.append(nn.Sequential(
                Conv(ch, ch, 3), Conv(ch, ch, 3),
                nn.Conv2d(ch, 4 * reg_max, 1)
            ))

        # ---- one2one head (lightweight — used for inference) ----
        # Shares most weights with one2many, but has separate final layers
        self.o_cls_branches = nn.ModuleList()
        self.o_reg_branches = nn.ModuleList()
        for ch in self.channels:
            self.o_cls_branches.append(nn.Sequential(
                Conv(ch, ch, 3), Conv(ch, ch, 3),
                nn.Conv2d(ch, num_classes, 1)
            ))
            self.o_reg_branches.append(nn.Sequential(
                Conv(ch, ch, 3), Conv(ch, ch, 3),
                nn.Conv2d(ch, 4 * reg_max, 1)
            ))

        self.dfl = DFL(reg_max)
        self._init_biases()

    def _init_biases(self):
        """Initialize biases for stable training start."""
        for branches in [self.m_cls_branches, self.o_cls_branches]:
            for branch in branches:
                nn.init.constant_(branch[-1].bias, -4.0)
        for branches in [self.m_reg_branches, self.o_reg_branches]:
            for branch in branches:
                nn.init.constant_(branch[-1].bias, 0.0)

    def forward(self, features):
        """
        Args:
            features: [N3, N4, N5] from neck

        Returns:
            If training:
                (one2many_cls, one2many_bbox), (one2one_cls, one2one_bbox)
            If eval:
                (one2one_cls, one2one_bbox)  — NMS-free output
        """
        m_cls, m_bbox = [], []
        o_cls, o_bbox = [], []

        for i, feat in enumerate(features):
            # one2many predictions (always computed during training)
            if self.training:
                mc = self.m_cls_branches[i](feat)
                mr = self.dfl(self.m_reg_branches[i](feat))
                m_cls.append(mc)
                m_bbox.append(mr)

            # one2one predictions (always computed)
            oc = self.o_cls_branches[i](feat)
            or_ = self.dfl(self.o_reg_branches[i](feat))
            o_cls.append(oc)
            o_bbox.append(or_)

        if self.training:
            return (m_cls, m_bbox), (o_cls, o_bbox)
        else:
            return o_cls, o_bbox


# =============================================================================
# Hungarian Matcher — one2one assignment
# =============================================================================

class HungarianMatcher:
    """
    Hungarian matching for one2one label assignment.

    For N predictions and M ground truths, find the optimal 1-to-1
    matching that minimizes the total cost:

      min Σ C(pred_i, gt_σ(i))  over permutations σ

    The cost C combines classification and localization:
      C(pred, gt) = λ_cls × L_cls + λ_box × L_box

    This is solved in O(N³) by the Hungarian algorithm (scipy).

    Reference: Carion et al., "DETR" (ECCV 2020) — Section 3.2
    """

    def __init__(self, num_classes=80, cls_weight=2.0, box_weight=5.0,
                 iou_weight=2.0):
        self.num_classes = num_classes
        self.cls_weight = cls_weight
        self.box_weight = box_weight
        self.iou_weight = iou_weight

    @torch.no_grad()
    def __call__(self, pred_scores, pred_bboxes, targets, num_gt):
        """
        Args:
            pred_scores: [B, N_total, num_classes] sigmoid scores
            pred_bboxes: [B, N_total, 4] decoded boxes
            targets: [num_gt_total, 6] (batch_idx, cls, cx, cy, w, h)
            num_gt: total number of ground truth boxes

        Returns:
            matched_indices: list of (pred_indices, gt_indices) per batch
        """
        batch_size = pred_scores.shape[0]
        device = pred_scores.device
        matched_indices = []

        for b in range(batch_size):
            gt_mask = targets[:, 0] == b
            if gt_mask.sum() == 0:
                matched_indices.append((
                    torch.tensor([], dtype=torch.long, device=device),
                    torch.tensor([], dtype=torch.long, device=device)
                ))
                continue

            gt_boxes = targets[gt_mask, 2:]   # [M, 4]
            gt_cls = targets[gt_mask, 1].long()  # [M]
            M = len(gt_cls)
            N = pred_scores.shape[1]

            # Cost matrix [N, M]
            # Classification cost: -log(pred_score[gt_class])
            cls_scores = pred_scores[b][:, gt_cls]  # [N, M]
            cls_cost = -torch.log(cls_scores.clamp(1e-8))

            # Box L1 cost
            pred_b = pred_bboxes[b].unsqueeze(1).expand(-1, M, -1)  # [N, M, 4]
            gt_b = gt_boxes.unsqueeze(0).expand(N, -1, -1)          # [N, M, 4]
            box_cost = (pred_b - gt_b).abs().sum(dim=-1)            # [N, M]

            # IoU cost
            iou_cost = -self._pairwise_iou(pred_bboxes[b], gt_boxes)  # [N, M]

            # Combined cost
            cost = (self.cls_weight * cls_cost +
                    self.box_weight * box_cost +
                    self.iou_weight * iou_cost)

            # Hungarian matching (scipy)
            cost_np = cost.cpu().numpy()
            pred_idx, gt_idx = linear_sum_assignment(cost_np)

            matched_indices.append((
                torch.tensor(pred_idx, dtype=torch.long, device=device),
                torch.tensor(gt_idx, dtype=torch.long, device=device)
            ))

        return matched_indices

    def _pairwise_iou(self, boxes1, boxes2, eps=1e-7):
        """Compute pairwise IoU between two sets of xywh boxes."""
        # Convert to xyxy
        b1_x1 = boxes1[:, 0:1] - boxes1[:, 2:3] / 2
        b1_y1 = boxes1[:, 1:2] - boxes1[:, 3:4] / 2
        b1_x2 = boxes1[:, 0:1] + boxes1[:, 2:3] / 2
        b1_y2 = boxes1[:, 1:2] + boxes1[:, 3:4] / 2

        b2_x1 = boxes2[:, 0:1] - boxes2[:, 2:3] / 2
        b2_y1 = boxes2[:, 1:2] - boxes2[:, 3:4] / 2
        b2_x2 = boxes2[:, 0:1] + boxes2[:, 2:3] / 2
        b2_y2 = boxes2[:, 1:2] + boxes2[:, 3:4] / 2

        # Broadcast: [N, 1] vs [1, M]
        inter_x1 = torch.max(b1_x1, b2_x1.T)
        inter_y1 = torch.max(b1_y1, b2_y1.T)
        inter_x2 = torch.min(b1_x2, b2_x2.T)
        inter_y2 = torch.min(b1_y2, b2_y2.T)

        inter = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
        area1 = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        area2 = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

        return inter / (area1 + area2.T - inter + eps)


# =============================================================================
# NMS-Free Post-Processing
# =============================================================================

def nms_free_postprocess(cls_scores, bbox_preds, conf_threshold=0.25, max_det=300):
    """
    Post-process one2one head predictions — NO NMS needed.

    The one2one head already produces non-overlapping predictions.
    We just need to threshold by confidence and limit detections.

    Args:
        cls_scores: list of [B, num_classes, H, W]
        bbox_preds: list of [B, 4, H, W]
        conf_threshold: minimum confidence to keep
        max_det: maximum detections per image

    Returns:
        list of (boxes, scores, class_ids) per image in batch
    """
    batch_size = cls_scores[0].shape[0]
    results = []

    # Flatten all scales
    all_cls = []
    all_bbox = []
    for cls, bbox in zip(cls_scores, bbox_preds):
        b, c, h, w = cls.shape
        all_cls.append(cls.permute(0, 2, 3, 1).reshape(b, -1, c))
        all_bbox.append(bbox.permute(0, 2, 3, 1).reshape(b, -1, 4))

    all_cls = torch.cat(all_cls, dim=1).sigmoid()    # [B, N, C]
    all_bbox = torch.cat(all_bbox, dim=1)              # [B, N, 4]

    for b in range(batch_size):
        # Get max class score per prediction
        scores, class_ids = all_cls[b].max(dim=-1)  # [N], [N]

        # Filter by confidence
        keep = scores > conf_threshold
        scores = scores[keep]
        class_ids = class_ids[keep]
        boxes = all_bbox[b][keep]

        # Limit to max_det
        if len(scores) > max_det:
            _, top_idx = scores.topk(max_det)
            scores = scores[top_idx]
            class_ids = class_ids[top_idx]
            boxes = boxes[top_idx]

        results.append((boxes, scores, class_ids))

    return results
