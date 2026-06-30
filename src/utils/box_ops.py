"""
Box operations and utilities.

Converts between different box formats and provides NMS.
"""

import torch


def xywh_to_xyxy(boxes):
    """
    Convert boxes from (cx, cy, w, h) to (x1, y1, x2, y2).

    Args:
        boxes: [..., 4] in (cx, cy, w, h) format

    Returns:
        [..., 4] in (x1, y1, x2, y2) format
    """
    cx, cy, w, h = boxes.chunk(4, dim=-1)
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return torch.cat([x1, y1, x2, y2], dim=-1)


def xyxy_to_xywh(boxes):
    """
    Convert boxes from (x1, y1, x2, y2) to (cx, cy, w, h).
    """
    x1, y1, x2, y2 = boxes.chunk(4, dim=-1)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return torch.cat([cx, cy, w, h], dim=-1)


def nms(boxes, scores, iou_threshold=0.65):
    """
    Non-Maximum Suppression — filter overlapping detections.

    Keeps the highest-scoring box and removes others that overlap too much.

    Args:
        boxes: [N, 4] in (x1, y1, x2, y2) format
        scores: [N] confidence scores
        iou_threshold: boxes with IoU > this are suppressed

    Returns:
        keep: indices of boxes to keep
    """
    if boxes.numel() == 0:
        return torch.zeros(0, dtype=torch.long, device=boxes.device)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    _, order = scores.sort(descending=True)

    keep = []
    while order.numel() > 0:
        i = order[0]
        keep.append(i)

        if order.numel() == 1:
            break

        # Compute IoU between the best box and all remaining boxes
        xx1 = torch.max(x1[i], x1[order[1:]])
        yy1 = torch.max(y1[i], y1[order[1:]])
        xx2 = torch.min(x2[i], x2[order[1:]])
        yy2 = torch.min(y2[i], y2[order[1:]])

        w = (xx2 - xx1).clamp(min=0)
        h = (yy2 - yy1).clamp(min=0)
        inter = w * h

        iou = inter / (areas[i] + areas[order[1:]] - inter)

        # Keep boxes with IoU below threshold
        keep_mask = iou <= iou_threshold
        order = order[1:][keep_mask]

    return torch.tensor(keep, device=boxes.device)


def scale_boxes(boxes, orig_size, input_size):
    """
    Scale boxes from model input size back to original image size.

    Args:
        boxes: [N, 4] in (x1, y1, x2, y2), normalized to input_size
        orig_size: (orig_h, orig_w)
        input_size: (input_h, input_w)

    Returns:
        scaled boxes in pixel coordinates
    """
    orig_h, orig_w = orig_size
    input_h, input_w = input_size

    # Calculate scale and padding
    gain = min(input_h / orig_h, input_w / orig_w)
    pad_w = (input_w - orig_w * gain) / 2
    pad_h = (input_h - orig_h * gain) / 2

    # Scale boxes
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] * input_w - pad_w) / gain
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] * input_h - pad_h) / gain

    # Clip to image bounds
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, orig_w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, orig_h)

    return boxes
