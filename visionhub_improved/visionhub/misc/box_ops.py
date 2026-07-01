# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Utilities for bounding box manipulation and GIoU.
"""
import torch, os
from torchvision.ops.boxes import box_area


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / (union + 1e-6)
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / (area + 1e-6)



# modified from torchvision to also return the union
def box_iou_pairwise(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, :2], boxes2[:, :2])  # [N,2]
    rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])  # [N,2]

    wh = (rb - lt).clamp(min=0)  # [N,2]
    inter = wh[:, 0] * wh[:, 1]  # [N]

    union = area1 + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou_pairwise(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    Input:
        - boxes1, boxes2: N,4
    Output:
        - giou: N, 4
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    assert boxes1.shape == boxes2.shape
    iou, union = box_iou_pairwise(boxes1, boxes2) # N, 4

    lt = torch.min(boxes1[:, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,2]
    area = wh[:, 0] * wh[:, 1]

    return iou - (area - union) / area


def complete_box_iou(boxes1, boxes2, eps=1e-7):
    """
    Complete IoU (CIoU) from https://arxiv.org/abs/1911.08287v1
    
    Improves upon GIoU by considering:
    - Overlap area
    - Distance between centers
    - Aspect ratio consistency
    
    The boxes should be in [x0, y0, x1, y1] format
    
    Returns a [N, M] pairwise matrix, where N = len(boxes1) and M = len(boxes2)
    """
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    
    iou, union = box_iou(boxes1, boxes2)
    
    # Center distance
    b1_x1, b1_y1, b1_x2, b1_y2 = boxes1[:, 0:1], boxes1[:, 1:2], boxes1[:, 2:3], boxes1[:, 3:4]
    b2_x1, b2_y1, b2_x2, b2_y2 = boxes2[:, 0:1], boxes2[:, 1:2], boxes2[:, 2:3], boxes2[:, 3:4]
    
    b1_cx = (b1_x1 + b1_x2) / 2
    b1_cy = (b1_y1 + b1_y2) / 2
    b2_cx = (b2_x1 + b2_x2) / 2
    b2_cy = (b2_y1 + b2_y2) / 2
    
    # Distance between box centers
    center_distance = (b1_cx - b2_cx.T) ** 2 + (b1_cy - b2_cy.T) ** 2
    
    # Diagonal of the smallest enclosing box
    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    diagonal_distance = wh[:, :, 0] ** 2 + wh[:, :, 1] ** 2 + eps
    
    # Aspect ratio penalty
    b1_w = b1_x2 - b1_x1
    b1_h = b1_y2 - b1_y1
    b2_w = b2_x2 - b2_x1
    b2_h = b2_y2 - b2_y1
    
    arctan_b1 = torch.atan(b1_w / (b1_h + eps))
    arctan_b2 = torch.atan(b2_w / (b2_h + eps))
    v = (4 / (torch.pi ** 2)) * torch.pow(arctan_b1 - arctan_b2.T, 2)
    
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)
    
    # CIoU
    ciou = iou - (center_distance / diagonal_distance + alpha * v)
    
    return ciou


def complete_box_iou_pairwise(boxes1, boxes2, eps=1e-7):
    """
    Complete IoU (CIoU) for pairwise boxes (N boxes compared element-wise)
    
    Input:
        - boxes1, boxes2: N,4 in [x0, y0, x1, y1] format
    Output:
        - ciou: N
    """
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    assert boxes1.shape == boxes2.shape
    
    iou, union = box_iou_pairwise(boxes1, boxes2)
    
    # Center coordinates
    b1_cx = (boxes1[:, 0] + boxes1[:, 2]) / 2
    b1_cy = (boxes1[:, 1] + boxes1[:, 3]) / 2
    b2_cx = (boxes2[:, 0] + boxes2[:, 2]) / 2
    b2_cy = (boxes2[:, 1] + boxes2[:, 3]) / 2
    
    # Distance between centers
    center_distance = (b1_cx - b2_cx) ** 2 + (b1_cy - b2_cy) ** 2
    
    # Diagonal of smallest enclosing box
    lt = torch.min(boxes1[:, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    diagonal_distance = wh[:, 0] ** 2 + wh[:, 1] ** 2 + eps
    
    # Aspect ratio
    b1_w = boxes1[:, 2] - boxes1[:, 0]
    b1_h = boxes1[:, 3] - boxes1[:, 1]
    b2_w = boxes2[:, 2] - boxes2[:, 0]
    b2_h = boxes2[:, 3] - boxes2[:, 1]
    
    v = (4 / (torch.pi ** 2)) * torch.pow(torch.atan(b1_w / (b1_h + eps)) - torch.atan(b2_w / (b2_h + eps)), 2)
    
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)
    
    ciou = iou - (center_distance / diagonal_distance + alpha * v)
    
    return ciou


def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)

    x_mask = (masks * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = (masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)

if __name__ == '__main__':
    x = torch.rand(5, 4)
    y = torch.rand(3, 4)
    iou, union = box_iou(x, y)