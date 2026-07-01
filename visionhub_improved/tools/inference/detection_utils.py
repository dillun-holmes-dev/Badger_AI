import numpy as np
import torch


def _pairwise_iou(box, boxes):
    if boxes.size == 0:
        return np.empty((0,), dtype=np.float32)

    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    inter_w = np.maximum(0.0, x2 - x1)
    inter_h = np.maximum(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    box_area = max(0.0, (box[2] - box[0]) * (box[3] - box[1]))
    boxes_area = np.maximum(0.0, (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))
    union = np.maximum(box_area + boxes_area - inter_area, 1e-6)

    return inter_area / union


def class_agnostic_nms_indices(boxes, scores, iou_threshold):
    """Keep only the highest-confidence box among strongly overlapping detections."""
    if iou_threshold is None or iou_threshold <= 0:
        return np.arange(len(scores), dtype=np.int64)

    boxes = np.asarray(boxes, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)

    if len(scores) <= 1:
        return np.arange(len(scores), dtype=np.int64)

    order = np.argsort(-scores, kind="stable")
    keep = []

    while order.size > 0:
        current = order[0]
        keep.append(current)

        rest = order[1:]
        if rest.size == 0:
            break

        ious = _pairwise_iou(boxes[current], boxes[rest])
        order = rest[ious <= iou_threshold]

    return np.asarray(keep, dtype=np.int64)


def class_aware_nms_indices(boxes, scores, labels, iou_threshold):
    """Apply NMS independently per class label without requiring torchvision."""
    if iou_threshold is None or iou_threshold <= 0:
        return np.arange(len(scores), dtype=np.int64)

    boxes = np.asarray(boxes, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    labels = np.asarray(labels)

    if len(scores) <= 1:
        return np.arange(len(scores), dtype=np.int64)

    keep = []
    for label in np.unique(labels):
        label_idx = np.where(labels == label)[0]
        label_keep = class_agnostic_nms_indices(boxes[label_idx], scores[label_idx], iou_threshold)
        keep.extend(label_idx[label_keep].tolist())

    # Preserve descending score order across classes, mirroring batched NMS behavior.
    keep = np.asarray(keep, dtype=np.int64)
    keep = keep[np.argsort(-scores[keep], kind="stable")]
    return keep


def _pairwise_iou_torch(box, boxes):
    if boxes.numel() == 0:
        return boxes.new_empty((0,))

    x1 = torch.maximum(box[0], boxes[:, 0])
    y1 = torch.maximum(box[1], boxes[:, 1])
    x2 = torch.minimum(box[2], boxes[:, 2])
    y2 = torch.minimum(box[3], boxes[:, 3])

    inter_w = (x2 - x1).clamp(min=0)
    inter_h = (y2 - y1).clamp(min=0)
    inter_area = inter_w * inter_h

    box_area = ((box[2] - box[0]).clamp(min=0) * (box[3] - box[1]).clamp(min=0))
    boxes_area = ((boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0))
    union = (box_area + boxes_area - inter_area).clamp(min=1e-6)
    return inter_area / union


def class_agnostic_nms_indices_torch(boxes, scores, iou_threshold):
    """Torch equivalent of class_agnostic_nms_indices for GPU-resident tensors."""
    if iou_threshold is None or iou_threshold <= 0:
        return torch.arange(len(scores), device=scores.device)

    if scores.numel() <= 1:
        return torch.arange(len(scores), device=scores.device)

    order = torch.argsort(scores, descending=True, stable=True)
    keep = []

    while order.numel() > 0:
        current = order[0]
        keep.append(current)

        rest = order[1:]
        if rest.numel() == 0:
            break

        ious = _pairwise_iou_torch(boxes[current], boxes[rest])
        order = rest[ious <= iou_threshold]

    return torch.stack(keep) if keep else torch.empty((0,), dtype=torch.long, device=scores.device)


def class_aware_nms_indices_torch(boxes, scores, labels, iou_threshold):
    """Torch equivalent of class_aware_nms_indices for GPU-resident tensors."""
    if iou_threshold is None or iou_threshold <= 0:
        return torch.arange(len(scores), device=scores.device)

    if scores.numel() <= 1:
        return torch.arange(len(scores), device=scores.device)

    keep = []
    for label in torch.unique(labels):
        label_idx = torch.nonzero(labels == label, as_tuple=False).flatten()
        label_keep = class_agnostic_nms_indices_torch(boxes[label_idx], scores[label_idx], iou_threshold)
        keep.append(label_idx[label_keep])

    if not keep:
        return torch.empty((0,), dtype=torch.long, device=scores.device)

    keep = torch.cat(keep)
    return keep[torch.argsort(scores[keep], descending=True, stable=True)]


def suppress_duplicate_detections(detections, iou_threshold):
    if iou_threshold is None or iou_threshold <= 0 or len(detections) <= 1:
        return detections

    boxes = np.asarray([det["box"] for det in detections], dtype=np.float32)
    scores = np.asarray([det["score"] for det in detections], dtype=np.float32)
    keep = class_agnostic_nms_indices(boxes, scores, iou_threshold)
    return [detections[idx] for idx in keep]
