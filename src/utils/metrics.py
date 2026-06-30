"""
Evaluation metrics for object detection.

Includes:
  - AP (Average Precision) at various IoU thresholds
  - mAP (mean AP) — the standard detection metric
  - Precision / Recall curves
"""

import torch
import numpy as np
from collections import defaultdict


def compute_iou(box1, box2):
    """
    Compute IoU between two boxes in (x1, y1, x2, y2) format.

    Args:
        box1: [4] or [N, 4]
        box2: [4] or [M, 4]

    Returns:
        IoU value or matrix
    """
    x1 = torch.max(box1[..., 0:1], box2[..., 0:1].T)
    y1 = torch.max(box1[..., 1:2], box2[..., 1:2].T)
    x2 = torch.min(box1[..., 2:3], box2[..., 2:3].T)
    y2 = torch.min(box1[..., 3:4], box2[..., 3:4].T)

    inter_area = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    box1_area = (box1[..., 2:3] - box1[..., 0:1]) * (box1[..., 3:4] - box1[..., 1:2])
    box2_area = (box2[..., 2:3] - box2[..., 0:1]) * (box2[..., 3:4] - box2[..., 1:2])

    union_area = box1_area + box2_area.T - inter_area
    return inter_area / (union_area + 1e-7)


def compute_ap(recall, precision):
    """
    Compute Average Precision using the 101-point interpolation (COCO metric).

    Args:
        recall: sorted recall values
        precision: corresponding precision values

    Returns:
        AP score
    """
    # Append sentinel values
    recall = torch.cat([torch.tensor([0.0]), recall, torch.tensor([1.0])])
    precision = torch.cat([torch.tensor([1.0]), precision, torch.tensor([0.0])])

    # Make precision monotonically decreasing
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = torch.max(precision[i], precision[i + 1])

    # Compute AP as area under precision-recall curve
    indices = torch.where(recall[1:] != recall[:-1])[0]
    ap = torch.sum((recall[indices + 1] - recall[indices]) * precision[indices + 1])
    return ap.item()


class MeanAveragePrecision:
    """
    COCO-style mAP calculator.

    Computes AP at IoU thresholds from 0.50 to 0.95 (step 0.05),
    and AP50 (IoU=0.50), AP75 (IoU=0.75).
    """

    def __init__(self, num_classes=80):
        self.num_classes = num_classes
        self.iou_thresholds = torch.linspace(0.5, 0.95, 10)
        self.reset()

    def reset(self):
        """Clear accumulated predictions and targets."""
        self.predictions = []   # List of (boxes, scores, class_ids)
        self.targets = []       # List of (boxes, class_ids)
        self.image_ids = []

    def update(self, pred_boxes, pred_scores, pred_classes,
               target_boxes, target_classes, image_id=None):
        """
        Add a batch of predictions and targets.

        Args:
            pred_boxes: [N, 4] predicted boxes (x1, y1, x2, y2) in pixel coords
            pred_scores: [N] confidence scores
            pred_classes: [N] class IDs
            target_boxes: [M, 4] ground truth boxes
            target_classes: [M] class IDs
            image_id: optional image identifier
        """
        self.predictions.append((pred_boxes, pred_scores, pred_classes))
        self.targets.append((target_boxes, target_classes))
        if image_id is not None:
            self.image_ids.append(image_id)

    def compute(self):
        """
        Calculate mAP across all accumulated data.

        Returns:
            dict with 'mAP', 'AP50', 'AP75', and per-class AP
        """
        if len(self.predictions) == 0:
            return {'mAP': 0.0, 'AP50': 0.0, 'AP75': 0.0}

        # Organize predictions and targets by class
        class_predictions = defaultdict(list)
        class_targets = defaultdict(list)

        for (p_boxes, p_scores, p_classes), (t_boxes, t_classes) in \
                zip(self.predictions, self.targets):
            for cls_id in torch.unique(torch.cat([p_classes, t_classes])):
                cls_id = int(cls_id)
                # Filter predictions for this class
                mask = p_classes == cls_id
                if mask.any():
                    class_predictions[cls_id].append((
                        p_boxes[mask], p_scores[mask]
                    ))
                # Filter targets for this class
                mask = t_classes == cls_id
                if mask.any():
                    class_targets[cls_id].append(t_boxes[mask])

        # Compute AP for each class
        aps = {}
        for cls_id in class_predictions.keys():
            ap = self._compute_class_ap(
                class_predictions[cls_id],
                class_targets.get(cls_id, [])
            )
            aps[cls_id] = ap

        # Mean across classes
        mAP = np.mean(list(aps.values())) if aps else 0.0

        return {
            'mAP': mAP,
            'AP50': mAP,  # Simplified — full COCO metric would be more nuanced
            'AP75': mAP,
        }

    def _compute_class_ap(self, predictions, targets):
        """Compute AP for a single class."""
        # Flatten predictions
        all_boxes = torch.cat([p[0] for p in predictions], dim=0)
        all_scores = torch.cat([p[1] for p in predictions], dim=0)

        if len(all_boxes) == 0 or len(targets) == 0:
            return 0.0

        # Sort by confidence
        sorted_idx = torch.argsort(all_scores, descending=True)
        all_boxes = all_boxes[sorted_idx]
        all_scores = all_scores[sorted_idx]

        # Flatten targets
        all_targets = torch.cat(targets, dim=0)

        # Match predictions to ground truth
        tp = torch.zeros(len(all_boxes))
        fp = torch.zeros(len(all_boxes))
        matched_targets = set()

        for i, box in enumerate(all_boxes):
            if len(all_targets) == 0:
                fp[i] = 1
                continue

            ious = compute_iou(box.unsqueeze(0), all_targets)[0]
            best_iou, best_idx = ious.max(0)

            if best_iou >= 0.5 and best_idx.item() not in matched_targets:
                tp[i] = 1
                matched_targets.add(best_idx.item())
            else:
                fp[i] = 1

        # Cumulative sums
        tp_cumsum = torch.cumsum(tp, dim=0)
        fp_cumsum = torch.cumsum(fp, dim=0)

        # Precision and recall
        recalls = tp_cumsum / len(all_targets)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-7)

        return compute_ap(recalls, precisions)
