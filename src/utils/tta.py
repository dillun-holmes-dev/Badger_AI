"""
Test-Time Augmentation (TTA) and Weighted Box Fusion (WBF).

TTA: Run inference on multiple augmented versions of the same image,
     then merge predictions. This is a free accuracy boost — no
     retraining needed, just more inference compute.

WBF: Better than NMS for merging overlapping predictions. Instead of
     discarding lower-confidence boxes, WBF fuses them into a weighted
     average, producing more precise boxes. Especially useful with TTA.

The combination (TTA + WBF) typically gives +0.5-1.5% mAP at the cost
of 4-8× inference time (depending on number of augmentations used).

Reference:
  - Solovyev et al., "Weighted Boxes Fusion" (arXiv:1910.13302)
  - TTA is standard practice (used by all top COCO challenge entries)
"""

import torch
import torch.nn.functional as F
import numpy as np


# =============================================================================
# 1. Test-Time Augmentation (TTA)
# =============================================================================

class TTAPredictor:
    """
    Apply test-time augmentations and merge predictions.

    Standard TTA transforms for object detection:
      1. Original image
      2. Horizontal flip
      3. Scale 0.8× (zoom out)
      4. Scale 1.2× (zoom in)

    Each augmented image is run through the model independently.
    Bounding boxes from flipped images are un-flipped.
    All predictions are merged via WBF (preferred) or NMS.

    Usage:
        tta = TTAPredictor(model, scales=[1.0, 0.8, 1.2], use_flip=True)
        boxes, scores, classes = tta(image)
    """

    def __init__(self, model, img_size=640, scales=None, use_flip=True,
                 conf_threshold=0.001, iou_threshold=0.65, max_det=300):
        self.model = model
        self.img_size = img_size
        self.scales = scales or [1.0, 0.83, 1.2]
        self.use_flip = use_flip
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_det = max_det

    @torch.no_grad()
    def __call__(self, image):
        """
        Args:
            image: [3, H, W] tensor or numpy array, normalized

        Returns:
            boxes: [N, 4] in (x1, y1, x2, y2) pixel coordinates
            scores: [N] confidence scores
            classes: [N] class IDs
        """
        all_boxes = []
        all_scores = []
        all_classes = []

        device = next(self.model.parameters()).device
        orig_h, orig_w = image.shape[-2:]

        for scale in self.scales:
            # Resize
            new_h = int(self.img_size * scale)
            new_w = int(self.img_size * scale)
            img_scaled = F.interpolate(
                image.unsqueeze(0), size=(new_h, new_w),
                mode='bilinear', align_corners=False
            ).to(device)

            # Original
            boxes, scores, classes = self._predict(img_scaled, orig_h, orig_w)
            all_boxes.append(boxes)
            all_scores.append(scores)
            all_classes.append(classes)

            # Flipped
            if self.use_flip:
                img_flipped = torch.flip(img_scaled, dims=[3])  # Flip width
                boxes_f, scores_f, classes_f = self._predict(img_flipped, orig_h, orig_w)

                # Un-flip boxes
                if len(boxes_f) > 0:
                    boxes_f[:, [0, 2]] = orig_w - boxes_f[:, [2, 0]]

                all_boxes.append(boxes_f)
                all_scores.append(scores_f)
                all_classes.append(classes_f)

        # Concatenate all predictions
        if all_boxes:
            all_boxes = torch.cat(all_boxes, dim=0)
            all_scores = torch.cat(all_scores, dim=0)
            all_classes = torch.cat(all_classes, dim=0)

            # WBF fusion
            boxes, scores, classes = weighted_box_fusion(
                all_boxes, all_scores, all_classes,
                iou_threshold=self.iou_threshold,
                conf_threshold=self.conf_threshold,
            )
        else:
            boxes = torch.zeros((0, 4))
            scores = torch.zeros((0,))
            classes = torch.zeros((0,), dtype=torch.long)

        # Limit to max_det
        if len(scores) > self.max_det:
            _, top_idx = scores.topk(self.max_det)
            boxes = boxes[top_idx]
            scores = scores[top_idx]
            classes = classes[top_idx]

        return boxes, scores, classes

    def _predict(self, image_batch, orig_h, orig_w):
        """Run model inference and scale boxes back to original image size."""
        # Model forward pass
        cls_scores, bbox_preds = self.model(image_batch)

        # Post-process: decode boxes + NMS (simplified — uses Badger's head)
        # In production, use the full post-processing pipeline
        # For now, return empty (placeholder)
        return (
            torch.zeros((0, 4), device=image_batch.device),
            torch.zeros((0,), device=image_batch.device),
            torch.zeros((0,), dtype=torch.long, device=image_batch.device),
        )


# =============================================================================
# 2. Weighted Box Fusion (WBF)
# =============================================================================

def weighted_box_fusion(boxes, scores, classes, iou_threshold=0.65,
                        conf_threshold=0.001, weights=None):
    """
    Weighted Box Fusion — merge overlapping predictions instead of discarding.

    Unlike NMS (which picks the highest score and discards the rest),
    WBF fuses all overlapping boxes into a weighted average box.
    This gives:
      - More precise localization (averaging reduces noise)
      - Better recall (no boxes are discarded, just merged)
      - Higher mAP when combined with TTA

    Algorithm:
      1. Sort predictions by confidence (descending)
      2. For each prediction:
         a. Find all existing fused clusters with IoU > threshold
         b. If found: add to the cluster with highest IoU
         c. If not found: create new cluster
      3. For each cluster: compute weighted average box and total score

    Args:
        boxes: [N, 4] in (x1, y1, x2, y2) format
        scores: [N] confidence scores
        classes: [N] class labels (integer IDs)
        iou_threshold: boxes with IoU > this are merged
        conf_threshold: minimum confidence to keep
        weights: optional per-box weights (default: scores)

    Returns:
        fused_boxes: [M, 4]
        fused_scores: [M]
        fused_classes: [M]
    """
    if len(boxes) == 0:
        return (
            torch.zeros((0, 4), device=boxes.device),
            torch.zeros((0,), device=scores.device),
            torch.zeros((0,), dtype=torch.long, device=classes.device),
        )

    if weights is None:
        weights = scores.clone()

    # Filter by confidence
    keep_mask = scores > conf_threshold
    if not keep_mask.any():
        return (
            torch.zeros((0, 4), device=boxes.device),
            torch.zeros((0,), device=scores.device),
            torch.zeros((0,), dtype=torch.long, device=classes.device),
        )

    boxes = boxes[keep_mask]
    scores = scores[keep_mask]
    classes = classes[keep_mask]
    weights = weights[keep_mask]

    # Sort by score descending
    sorted_idx = torch.argsort(scores, descending=True)
    boxes = boxes[sorted_idx]
    scores = scores[sorted_idx]
    classes = classes[sorted_idx]
    weights = weights[sorted_idx]

    # Clusters: each is (sum_weighted_boxes, sum_weights, class_id)
    clusters = []  # List of (boxes_list, weights_list, class_id)

    for i in range(len(boxes)):
        box = boxes[i:i+1]
        score_i = scores[i]
        cls_i = int(classes[i])
        weight_i = weights[i]

        # Find best matching cluster (same class, highest IoU)
        best_cluster_idx = -1
        best_iou = iou_threshold

        for j, (cluster_boxes, cluster_weights, cluster_cls) in enumerate(clusters):
            if cluster_cls != cls_i:
                continue

            # Compute IoU with the cluster's representative box
            # (weighted average of all boxes in the cluster so far)
            cluster_rep = _weighted_average_box(cluster_boxes, cluster_weights)
            iou = _box_iou(box, cluster_rep.unsqueeze(0)).item()

            if iou > best_iou:
                best_iou = iou
                best_cluster_idx = j

        if best_cluster_idx >= 0:
            # Add to existing cluster
            clusters[best_cluster_idx][0].append(box)
            clusters[best_cluster_idx][1].append(weight_i)
        else:
            # Create new cluster
            clusters.append(([box], [weight_i], cls_i))

    # Fuse each cluster
    fused_boxes = []
    fused_scores = []
    fused_classes = []

    for cluster_boxes, cluster_weights, cluster_cls in clusters:
        if len(cluster_boxes) == 1:
            fused_boxes.append(cluster_boxes[0])
            fused_scores.append(cluster_weights[0])
        else:
            # Weighted average of all boxes in the cluster
            boxes_stacked = torch.cat(cluster_boxes, dim=0)  # [K, 4]
            w_stacked = torch.tensor(cluster_weights, device=boxes.device)  # [K]

            # Weighted average
            fused_box = (boxes_stacked * w_stacked.view(-1, 1)).sum(dim=0)
            fused_box = fused_box / w_stacked.sum()

            # Fused score = average of scores × (1 - variance penalty)
            # Lower variance = higher confidence in the fusion
            mean_score = w_stacked.mean()
            # Penalty: if boxes disagree a lot, reduce confidence
            variance_penalty = 1.0 - _box_variance(boxes_stacked)
            fused_score = mean_score * variance_penalty

            fused_boxes.append(fused_box.unsqueeze(0))
            fused_scores.append(fused_score)

        fused_classes.append(cluster_cls)

    if fused_boxes:
        fused_boxes = torch.cat(fused_boxes, dim=0)
        fused_scores = torch.tensor(fused_scores, device=boxes.device)
        fused_classes = torch.tensor(fused_classes, device=boxes.device, dtype=torch.long)
    else:
        fused_boxes = torch.zeros((0, 4), device=boxes.device)
        fused_scores = torch.zeros((0,), device=boxes.device)
        fused_classes = torch.zeros((0,), dtype=torch.long, device=boxes.device)

    return fused_boxes, fused_scores, fused_classes


def _box_iou(box_a, box_b):
    """Pairwise IoU between two sets of boxes."""
    x1 = torch.max(box_a[:, 0], box_b[:, 0])
    y1 = torch.max(box_a[:, 1], box_b[:, 1])
    x2 = torch.min(box_a[:, 2], box_b[:, 2])
    y2 = torch.min(box_a[:, 3], box_b[:, 3])

    inter = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    area_a = (box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])
    area_b = (box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1])

    return inter / (area_a + area_b - inter + 1e-7)


def _weighted_average_box(boxes_list, weights_list):
    """Compute weighted average of multiple boxes."""
    if len(boxes_list) == 1:
        return boxes_list[0].squeeze(0)
    boxes = torch.cat(boxes_list, dim=0)
    weights = torch.tensor(weights_list, device=boxes.device).view(-1, 1)
    return (boxes * weights).sum(dim=0) / weights.sum()


def _box_variance(boxes):
    """Measure how much boxes in a cluster disagree (0-1, lower=better)."""
    if len(boxes) < 2:
        return 0.0
    # Compute pairwise IoU inconsistency
    ious = _box_iou(boxes, boxes)
    # Remove self-IoU (diagonal)
    mask = ~torch.eye(len(boxes), dtype=torch.bool, device=boxes.device)
    mean_iou = ious[mask].mean().item()
    # 1.0 = perfect agreement, 0.0 = no agreement
    return max(0.0, min(1.0, mean_iou))


# =============================================================================
# 3. Model Ensembling
# =============================================================================

class ModelEnsemble:
    """
    Ensemble multiple trained models for higher accuracy.

    Unlike distillation (which trains a student), ensembling combines
    predictions from multiple independently trained models at inference
    time. This is standard practice in competition settings.

    The ensemble typically gives +1-3% mAP over any single model.

    Usage:
        models = [model1, model2, model3]
        ensemble = ModelEnsemble(models)
        boxes, scores, classes = ensemble(image)
    """

    def __init__(self, models, weights=None, fusion='wbf'):
        """
        Args:
            models: list of trained Badger models
            weights: relative weight for each model (default: equal)
            fusion: 'wbf' or 'nms' — how to merge predictions
        """
        self.models = models
        self.weights = weights or [1.0] * len(models)
        self.fusion = fusion

        # Freeze all models
        for model in self.models:
            for p in model.parameters():
                p.requires_grad = False
            model.eval()

    @torch.no_grad()
    def __call__(self, image, conf_threshold=0.001, iou_threshold=0.65):
        """
        Run all models and merge predictions.

        Args:
            image: [3, H, W] normalized tensor
            conf_threshold: minimum confidence
            iou_threshold: IoU threshold for merging

        Returns:
            boxes, scores, classes
        """
        device = next(self.models[0].parameters()).device
        image = image.unsqueeze(0).to(device)

        all_boxes = []
        all_scores = []
        all_classes = []

        for model, weight in zip(self.models, self.weights):
            # Model forward pass (placeholder — needs full post-processing)
            cls_scores, bbox_preds = model(image)
            # TODO: full decode + scale back

        # Merge via WBF or NMS
        if len(all_boxes) > 0:
            all_boxes = torch.cat(all_boxes, dim=0)
            all_scores = torch.cat(all_scores, dim=0)
            all_classes = torch.cat(all_classes, dim=0)

            if self.fusion == 'wbf':
                boxes, scores, classes = weighted_box_fusion(
                    all_boxes, all_scores, all_classes,
                    iou_threshold=iou_threshold
                )
            else:
                from src.utils.box_ops import nms
                # NMS (simplified — WBF is better)
                keep = nms(all_boxes, all_scores, iou_threshold)
                boxes = all_boxes[keep]
                scores = all_scores[keep]
                classes = all_classes[keep]
        else:
            boxes = torch.zeros((0, 4))
            scores = torch.zeros((0,))
            classes = torch.zeros((0,), dtype=torch.long)

        return boxes, scores, classes
