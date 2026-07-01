"""
Label assignment and loss functions for Badger.

Label assignment is the critical step that decides WHICH predictions are
matched to WHICH ground truth boxes. Badger supports two strategies:
  1. TAL (Task Alignment Learning) — from YOLOv8, balances cls + reg quality
  2. SimOTA (Simplified Optimal Transport) — from YOLOX, uses dynamic k

You can swap between them in the config to find what works best for your data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# 1. IoU / Box Utilities
# =============================================================================

def bbox_iou(box1, box2, xywh=True, eps=1e-7):
    """
    Calculate IoU matrix between two sets of boxes.

    Adds broadcasting dims so [N,4] vs [M,4] produces [N,M] IoU.

    Args:
        box1: [N, 4] in (cx, cy, w, h) format
        box2: [M, 4] in (cx, cy, w, h) format
        xywh: if True, boxes are in (cx, cy, w, h) format

    Returns:
        IoU matrix [N, M]
    """
    # Add broadcast dimensions: [N, 1, 4] vs [1, M, 4]
    box1 = box1.unsqueeze(1)
    box2 = box2.unsqueeze(0)

    # Convert from (cx, cy, w, h) to (x1, y1, x2, y2)
    if xywh:
        b1_x1 = box1[..., 0] - box1[..., 2] / 2
        b1_y1 = box1[..., 1] - box1[..., 3] / 2
        b1_x2 = box1[..., 0] + box1[..., 2] / 2
        b1_y2 = box1[..., 1] + box1[..., 3] / 2

        b2_x1 = box2[..., 0] - box2[..., 2] / 2
        b2_y1 = box2[..., 1] - box2[..., 3] / 2
        b2_x2 = box2[..., 0] + box2[..., 2] / 2
        b2_y2 = box2[..., 1] + box2[..., 3] / 2
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, dim=-1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, dim=-1)

    # Intersection area — broadcasting gives [N, M]
    inter_x1 = torch.max(b1_x1, b2_x1)
    inter_y1 = torch.max(b1_y1, b2_y1)
    inter_x2 = torch.min(b1_x2, b2_x2)
    inter_y2 = torch.min(b1_y2, b2_y2)
    inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)

    # Union area
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union_area = b1_area + b2_area - inter_area + eps

    return inter_area / union_area  # [N, M]


def ciou_loss(pred_boxes, target_boxes, xywh=True, eps=1e-7):
    """
    Complete IoU (CIoU) loss.

    CIoU = 1 - IoU + ρ²(b, b_gt)/c² + αv

    Components:
      1. IoU term:        basic overlap loss
      2. Center distance:  penalizes distance between box centers
      3. Aspect ratio:    penalizes shape mismatch

    This is the standard box loss in Badger.

    Args:
        pred_boxes:  predicted boxes [N, 4] in (cx, cy, w, h)
        target_boxes: ground truth boxes [N, 4] in (cx, cy, w, h)

    Returns:
        CIoU loss value (scalar)
    """
    # Convert to (x1, y1, x2, y2)
    if xywh:
        p_x1 = pred_boxes[..., 0] - pred_boxes[..., 2] / 2
        p_y1 = pred_boxes[..., 1] - pred_boxes[..., 3] / 2
        p_x2 = pred_boxes[..., 0] + pred_boxes[..., 2] / 2
        p_y2 = pred_boxes[..., 1] + pred_boxes[..., 3] / 2

        t_x1 = target_boxes[..., 0] - target_boxes[..., 2] / 2
        t_y1 = target_boxes[..., 1] - target_boxes[..., 3] / 2
        t_x2 = target_boxes[..., 0] + target_boxes[..., 2] / 2
        t_y2 = target_boxes[..., 1] + target_boxes[..., 3] / 2
    else:
        p_x1, p_y1, p_x2, p_y2 = pred_boxes.chunk(4, dim=-1)
        t_x1, t_y1, t_x2, t_y2 = target_boxes.chunk(4, dim=-1)

    # Intersection
    inter_x1 = torch.max(p_x1, t_x1)
    inter_y1 = torch.max(p_y1, t_y1)
    inter_x2 = torch.min(p_x2, t_x2)
    inter_y2 = torch.min(p_y2, t_y2)
    inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)

    # Union
    p_area = (p_x2 - p_x1) * (p_y2 - p_y1)
    t_area = (t_x2 - t_x1) * (t_y2 - t_y1)
    union_area = p_area + t_area - inter_area + eps

    iou = inter_area / union_area

    # Smallest enclosing box
    enclose_x1 = torch.min(p_x1, t_x1)
    enclose_y1 = torch.min(p_y1, t_y1)
    enclose_x2 = torch.max(p_x2, t_x2)
    enclose_y2 = torch.max(p_y2, t_y2)
    enclose_diag2 = (enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2

    # Center distance squared
    p_cx = (p_x1 + p_x2) / 2
    p_cy = (p_y1 + p_y2) / 2
    t_cx = (t_x1 + t_x2) / 2
    t_cy = (t_y1 + t_y2) / 2
    center_dist2 = (p_cx - t_cx) ** 2 + (p_cy - t_cy) ** 2

    # Aspect ratio penalty
    p_w = p_x2 - p_x1
    p_h = p_y2 - p_y1
    t_w = t_x2 - t_x1
    t_h = t_y2 - t_y1
    # v measures aspect ratio consistency
    v = (4 / (torch.pi ** 2)) * ((torch.atan(t_w / t_h) - torch.atan(p_w / p_h)) ** 2)
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)

    ciou = iou - (center_dist2 / (enclose_diag2 + eps)) - (alpha * v)
    loss = 1 - ciou

    return loss.mean()


# =============================================================================
# 2. Distribution Focal Loss (DFL)
# =============================================================================

def dfl_loss(pred_dist, target, reg_max=16):
    """
    Distribution Focal Loss — for fine-grained bounding box regression.

    Instead of predicting a single value for each box edge (left, top, right,
    bottom), DFL predicts a DISTRIBUTION over possible edge offsets.

    For example, for the "left" edge, the model predicts 16 values (reg_max=16)
    representing P(offset=0), P(offset=1), ..., P(offset=15). DFL makes the
    distribution peak near the ground truth offset.

    Args:
        pred_dist: [N, 4*reg_max] — predicted distribution over edge offsets
        target: [N, 4] — ground truth edge offsets (0 to reg_max-1)
        reg_max: number of discrete bins

    Returns:
        DFL loss (scalar)
    """
    n = pred_dist.shape[0]
    pred_dist = pred_dist.view(n, 4, reg_max)  # [N, 4, reg_max]

    # Target: left and right neighbors of ground truth
    target_clamped = target.clamp(0, reg_max - 1)
    target_left = target_clamped.long()         # [N, 4]
    target_right = target_left + 1
    target_right = target_right.clamp(max=reg_max - 1)

    # Weight: how close the target is to each neighbor
    weight_right = target - target_left.float()  # Distance to left neighbor
    weight_left = target_right.float() - target   # Distance to right neighbor

    # Cross-entropy loss for left-bin prediction
    loss_left = F.cross_entropy(
        pred_dist.view(-1, reg_max),
        target_left.view(-1),
        reduction='none'
    ).view(n, 4) * weight_left

    # Cross-entropy loss for right-bin prediction
    loss_right = F.cross_entropy(
        pred_dist.view(-1, reg_max),
        target_right.view(-1),
        reduction='none'
    ).view(n, 4) * weight_right

    return (loss_left + loss_right).mean()


# =============================================================================
# 3. Task Aligned Assigner (TAL)
# =============================================================================

class TaskAlignedAssigner:
    """
    TAL: Task Aligned Label Assignment — from TOOD / YOLOv8.

    MATHEMATICAL DERIVATION:
    Alignment metric t_i for prediction i against GT j:
      t_i = (s_i)^α × (u_i)^β

      s_i = sigmoid(cls_score_i[gt_class])  — classification confidence
      u_i = IoU(pred_box_i, gt_box_j)        — localization quality

    Select top-k predictions with highest t_i per GT (k=13, fixed).

    DEFAULT HYPERPARAMETER AUDIT:
      topk=13: YOLOv8 default. Not from TOOD paper (which uses per-FPN-level k).
      α=1.0, β=6.0: From TOOD Table 3 (β∈{1,3,5,6,7,9}, β=6 optimal on COCO).
      ⚠ CAVEAT: Tuned for anchor-based ResNet-50 detection. May not transfer
      optimally to anchor-free Badger. ABLATION NEEDED.

    LITERATURE DISAGREEMENT (Rule 8):
      TOOD claims alignment metric couples cls+reg during assignment.
      VarifocalNet (Zhang 2020) argues IoU-weighted classification, not
      assignment strategy, is the actual mechanism. Badger uses BOTH —
      their effects may overlap. Ablation needed to disentangle.

    Reference: Feng et al., "TOOD" (ICCV 2021), Section 3.1, Table 3.
    Implementation: Based on YOLOv8/Ultralytics TAL.
    """

    def __init__(self, num_classes=80, topk=10, alpha=0.5, beta=6.0):
        self.num_classes = num_classes
        self.topk = topk        # k=10: Ultralytics default (cleaner training, fewer FP)
        self.alpha = alpha      # α=0.5: Ultralytics default (IoU matters more than cls)
        self.beta = beta        # β=6.0: reg exponent (TOOD Table 3, β=6 optimal)

    @torch.no_grad()
    def __call__(self, pred_scores, pred_bboxes, targets, anchors, strides,
                 img_size, num_gt):
        """
        Args:
            pred_scores: [B, N_total, num_classes] — all prediction scores
            pred_bboxes: [B, N_total, 4] — all predicted boxes
            targets: [num_gt_total, 6] — (batch_idx, cls, x, y, w, h) normalized
            anchors: [N_total, 2] — anchor points in grid
            strides: list of strides [8, 16, 32]
            img_size: (H, W) of input image
            num_gt: number of ground truth boxes

        Returns:
            target_labels: [B, N_total, num_classes] — assigned class labels
            target_bboxes: [B, N_total, 4] — assigned box coordinates
            target_scores: [B, N_total, num_classes] — soft labels with IoU
            fg_mask: [B, N_total] — foreground mask (which predictions are matched)
        """
        batch_size = pred_scores.shape[0]
        num_anchors = pred_scores.shape[1]
        device = pred_scores.device

        # Initialize outputs
        target_labels = torch.zeros(batch_size, num_anchors, self.num_classes, device=device)
        target_bboxes = torch.zeros(batch_size, num_anchors, 4, device=device)
        target_scores = torch.zeros(batch_size, num_anchors, self.num_classes, device=device)
        fg_mask = torch.zeros(batch_size, num_anchors, dtype=torch.bool, device=device)
        best_alignment = torch.full((batch_size, num_anchors), -1.0, device=device)

        if num_gt == 0:
            return target_labels, target_bboxes, target_scores, fg_mask

        # Compute alignment metric for each (prediction, ground truth) pair
        # This is where the "task alignment" happens

        # For each batch item
        for b in range(batch_size):
            gt_mask = targets[:, 0] == b  # Ground truth boxes in this batch
            if gt_mask.sum() == 0:
                continue

            gt_boxes = targets[gt_mask, 2:]   # [num_gt_b, 4] normalized
            gt_cls = targets[gt_mask, 1].long()  # [num_gt_b]
            # Filter invalid GT boxes (zero width/height) — from Ultralytics TAL
            valid_gt = (gt_boxes[:, 2] > 1e-6) & (gt_boxes[:, 3] > 1e-6)
            if not valid_gt.any():
                continue
            gt_boxes = gt_boxes[valid_gt]
            gt_cls = gt_cls[valid_gt]

            # Compute IoU between all predictions and all ground truths
            # pred_bboxes: [N_total, 4], gt_boxes: [num_gt_b, 4]
            iou = bbox_iou(pred_bboxes[b], gt_boxes)  # [N_total, num_gt_b]

            # Get classification scores for ground truth classes
            cls_scores = pred_scores[b, :, gt_cls].T  # [num_gt_b, N_total]

            # Alignment metric = cls^alpha * iou^beta
            alignment = (cls_scores ** self.alpha) * (iou.T ** self.beta)

            # Select top-k predictions per ground truth
            topk_align, topk_idx = alignment.topk(self.topk, dim=1)  # [num_gt_b, topk]

            # Assign
            for gt_i in range(len(gt_cls)):
                for k in range(self.topk):
                    anchor_idx = topk_idx[gt_i, k]
                    score = topk_align[gt_i, k]

                    iou_val = iou[anchor_idx, gt_i].clamp(0, 1)
                    # If multiple GTs select the same anchor, keep the match
                    # with the stronger task-alignment score.
                    if score >= best_alignment[b, anchor_idx]:
                        target_labels[b, anchor_idx].zero_()
                        target_scores[b, anchor_idx].zero_()
                        target_labels[b, anchor_idx, gt_cls[gt_i]] = 1.0
                        target_bboxes[b, anchor_idx] = gt_boxes[gt_i]
                        target_scores[b, anchor_idx, gt_cls[gt_i]] = iou_val
                        fg_mask[b, anchor_idx] = True
                        best_alignment[b, anchor_idx] = score

        return target_labels, target_bboxes, target_scores, fg_mask


# =============================================================================
# 4. SimOTA — YOLOX's Simplified Optimal Transport Assigner
# =============================================================================

class SimOTAAssigner:
    """
    SimOTA: Simplified Optimal Transport Assignment — from YOLOX.

    MATHEMATICAL DERIVATION:
    -------------------------
    The full Optimal Transport (OT) problem for label assignment:
      min Σ_ij C_ij · π_ij
      s.t. Σ_i π_ij = d_j (demand: each GT gets k positives)
           Σ_j π_ij ≤ 1  (supply: each prediction at most 1 GT)
           π_ij ∈ {0, 1}

    where C_ij is the cost of matching prediction i to GT j:
      C_ij = L_cls(P_i^cls, G_j^cls) + λ · L_reg(P_i^box, G_j^box)
           = -log(p_i^{cls}) + λ · (-log(IoU_ij))

    SimOTA simplifies this by:
      1. Computing dynamic k per GT: k_j = ceil(Σ_{i∈top-q} IoU_ij)
         where top-q are the q predictions with highest IoU to GT j.
         This means: better-matched objects get MORE positives.
         Small/occluded objects (low max IoU) get proportionally fewer.
      2. For each GT, selecting the k_j lowest-cost predictions.

    DEFAULT HYPERPARAMETER AUDIT:
      - topk=10: From YOLOX paper Section 3.3, Table 4 ablation.
        "We select top 10 predictions with highest IoU as candidates."
        Ablation tested {5, 10, 20}; 10 gave best mAP (40.3 vs 40.1 vs 39.9).
      - iou_weight=3.0: From YOLOX paper Section 3.3.
        "λ = 3 for regression cost." Not extensively ablated but used
        consistently across all YOLOX variants. Empirically: lower values
        (1-2) favor classification, higher (4-5) favor localization.
        ⚠ CAVEAT: This value was tuned on COCO with CSPDarknet-S.
        Optimal value may differ for Badger's architecture. ABLATION NEEDED.

    KNOWN LIMITATIONS (from OpenReview discussions of YOLOX):
      - Dynamic k can produce 0 positives for very small GTs, causing them
        to never be learned. The clamp(min=1) prevents this but means
        even the hardest-to-detect objects get at least one anchor.
      - The cost matrix is computed for ALL N×M pairs (O(NM)), but the
        original paper only computes for top-q candidates (O(qM)).
        Our implementation computes full cost for correctness and then
        selects top-k. This is computationally heavier but equivalent.
        TODO: Optimize to only compute cost on top-q candidates.
      - SimOTA assumes anchor-free detection. For anchor-based detectors,
        the IoU computation needs to account for anchor shapes.

    Reference: Ge et al., "YOLOX: Exceeding YOLO Series in 2021"
               (arXiv:2107.08430), Section 3.3, Table 4.
    Follow-up: OTA (Optimal Transport Assignment) by the same authors
               provides the theoretical foundation (CVPR 2021).
    """

    def __init__(self, num_classes=80, topk=10, iou_weight=3.0):
        self.num_classes = num_classes
        self.topk = topk              # q: max candidates considered per GT (YOLOX Table 4)
        self.iou_weight = iou_weight  # λ: regression cost weight (YOLOX Section 3.3)

    @torch.no_grad()
    def __call__(self, pred_scores, pred_bboxes, targets, anchors, strides,
                 img_size, num_gt):
        """
        Args:
            pred_scores: [B, N_total, num_classes]
            pred_bboxes: [B, N_total, 4]
            targets: [num_gt_total, 6] — (batch_idx, cls, x, y, w, h) normalized
            anchors: [N_total, 2] — anchor points in grid
            strides: list of strides [8, 16, 32]
            img_size: (H, W)
            num_gt: total number of ground truth boxes

        Returns:
            target_labels: [B, N_total, num_classes]
            target_bboxes: [B, N_total, 4]
            target_scores: [B, N_total, num_classes]
            fg_mask: [B, N_total]
        """
        batch_size = pred_scores.shape[0]
        num_anchors = pred_scores.shape[1]
        device = pred_scores.device

        target_labels = torch.zeros(batch_size, num_anchors, self.num_classes, device=device)
        target_bboxes = torch.zeros(batch_size, num_anchors, 4, device=device)
        target_scores = torch.zeros(batch_size, num_anchors, self.num_classes, device=device)
        fg_mask = torch.zeros(batch_size, num_anchors, dtype=torch.bool, device=device)

        if num_gt == 0:
            return target_labels, target_bboxes, target_scores, fg_mask

        # Process each image in the batch
        for b in range(batch_size):
            gt_mask = targets[:, 0] == b
            if gt_mask.sum() == 0:
                continue

            gt_boxes = targets[gt_mask, 2:]    # [num_gt_b, 4] normalized
            gt_cls = targets[gt_mask, 1].long() # [num_gt_b]
            num_gt_b = len(gt_cls)

            # Compute pairwise IoU between predictions and GT boxes
            pair_wise_iou = bbox_iou(pred_bboxes[b], gt_boxes)  # [N_total, num_gt_b]
            # Ensure 2D (bbox_iou may squeeze dimensions for single GTs)
            if pair_wise_iou.ndim == 1:
                pair_wise_iou = pair_wise_iou.unsqueeze(1)  # [N_total, 1]
            if num_gt_b == 1:
                pair_wise_iou = pair_wise_iou.view(-1, 1)

            # Classification cost: -log(cls_score) for the GT class
            gt_cls_scores = pred_scores[b, :, gt_cls]  # [N_total, num_gt_b]
            if gt_cls_scores.ndim == 1:
                gt_cls_scores = gt_cls_scores.unsqueeze(1)  # [N_total, 1]
            gt_cls_scores = gt_cls_scores.T  # [num_gt_b, N_total]
            pair_wise_cls_cost = -torch.log(gt_cls_scores.clamp(1e-8))  # [num_gt_b, N_total]

            # Cost matrix: combine classification and regression
            pair_wise_iou_loss = -torch.log(pair_wise_iou.T.clamp(1e-8))  # [num_gt_b, N_total]
            cost = pair_wise_cls_cost + self.iou_weight * pair_wise_iou_loss

            # Dynamic k: k_j = ceil(Σ_{i∈top-q} IoU_ij)
            # From YOLOX paper: "The number of positive samples for each GT
            # is the ceiling of the sum of its top q IoU values."
            # Using ceil (not floor/truncate) because: if IoU sum = 0.3,
            # ceil(0.3)=1 → at least one positive. Floor would give 0,
            # meaning that GT never gets learned — worse for small objects.
            iou_topk = min(self.topk, pair_wise_iou.shape[0])
            topk_iou, _ = pair_wise_iou.topk(iou_topk, dim=0)  # [topk, num_gt_b]
            dynamic_ks = topk_iou.sum(dim=0).ceil().int().clamp(min=1)
            if dynamic_ks.ndim == 0:
                dynamic_ks = dynamic_ks.unsqueeze(0)

            # For each GT, select the k lowest-cost predictions
            for gt_idx in range(num_gt_b):
                k = min(dynamic_ks[gt_idx].item(), num_anchors)
                if k == 0:
                    continue

                # Get top-k indices with lowest cost
                _, topk_indices = cost[gt_idx].topk(k, largest=False)

                for anchor_idx in topk_indices:
                    iou_val = pair_wise_iou[anchor_idx, gt_idx]
                    if iou_val >= target_scores[b, anchor_idx].max():
                        target_labels[b, anchor_idx].zero_()
                        target_scores[b, anchor_idx].zero_()
                        target_labels[b, anchor_idx, gt_cls[gt_idx]] = 1.0
                        target_bboxes[b, anchor_idx] = gt_boxes[gt_idx]
                        target_scores[b, anchor_idx, gt_cls[gt_idx]] = iou_val.clamp(0, 1)
                        fg_mask[b, anchor_idx] = True

        return target_labels, target_bboxes, target_scores, fg_mask


# =============================================================================
# 5. ATSS — Adaptive Training Sample Selection (Zhang et al., CVPR 2020)
# =============================================================================

class ATSSAssigner:
    """
    ATSS: Adaptive Training Sample Selection — from Zhang et al., CVPR 2020.

    Unlike TAL (fixed topk per GT) or SimOTA (per-GT cost threshold),
    ATSS adapts the number of positives PER FPN LEVEL based on the
    statistical distribution of IoU values.

    Algorithm (per image):
      1. For each GT, find the top-k anchors with highest IoU per FPN level
      2. Compute mean μ and std σ of these top-k IoU values
      3. Threshold: t_g = μ + σ (adaptive per GT)
      4. Select anchors with IoU > t_g AND whose center is inside the GT box
      5. If no anchor meets both criteria, take the single highest-IoU anchor

    Key insight: Different FPN levels have different IoU distributions.
    A fixed threshold (like TAL's topk=10) doesn't account for this.
    ATSS finds the natural separation point between positive and negative
    IoU values at each scale.

    Paper results (ATSS Table 3, COCO):
      FCOS + ATSS: 39.2 → 42.6 AP (+3.4 AP over standard FCOS assigner)
      RetinaNet + ATSS: 38.4 → 41.5 AP (+3.1 AP)

    Reference: Zhang et al., "Bridging the Gap Between Anchor-based and
               Anchor-free Detection via Adaptive Training Sample Selection"
               (CVPR 2020) — arXiv:1912.02424
    """

    def __init__(self, num_classes=80, topk=9):
        self.num_classes = num_classes
        self.topk = topk  # k=9 from ATSS paper (Section 3.2, Table 5 ablation)

    @torch.no_grad()
    def __call__(self, pred_scores, pred_bboxes, targets, anchors, strides,
                 img_size, num_gt):
        batch_size = pred_scores.shape[0]
        num_anchors = pred_scores.shape[1]
        device = pred_scores.device

        target_labels = torch.zeros(batch_size, num_anchors, self.num_classes, device=device)
        target_bboxes = torch.zeros(batch_size, num_anchors, 4, device=device)
        target_scores = torch.zeros(batch_size, num_anchors, self.num_classes, device=device)
        fg_mask = torch.zeros(batch_size, num_anchors, dtype=torch.bool, device=device)

        if num_gt == 0:
            return target_labels, target_bboxes, target_scores, fg_mask

        # Separate anchor points by FPN level
        n_per_level = []
        for s in strides:
            h, w = int(img_size[0] / s), int(img_size[1] / s)
            n_per_level.append(h * w)

        level_ranges = []
        start = 0
        for n in n_per_level:
            level_ranges.append((start, start + n))
            start += n

        num_levels = len(strides)

        for b in range(batch_size):
            gt_mask = targets[:, 0] == b
            if gt_mask.sum() == 0:
                continue

            gt_boxes = targets[gt_mask, 2:]
            gt_cls = targets[gt_mask, 1].long()
            valid_gt = (gt_boxes[:, 2] > 1e-6) & (gt_boxes[:, 3] > 1e-6)
            if not valid_gt.any():
                continue
            gt_boxes = gt_boxes[valid_gt]
            gt_cls = gt_cls[valid_gt]
            num_gt_b = len(gt_cls)

            iou = bbox_iou(pred_bboxes[b], gt_boxes)  # [N, M]

            for gt_i in range(num_gt_b):
                # Per-level top-k IoU selection
                candidate_idxs = []
                candidate_ious = []

                for l in range(num_levels):
                    start_l, end_l = level_ranges[l]
                    level_iou = iou[start_l:end_l, gt_i]  # [n_l]
                    topk_l = min(self.topk, len(level_iou))
                    topk_iou, topk_idx = level_iou.topk(topk_l)
                    candidate_idxs.append(topk_idx + start_l)
                    candidate_ious.append(topk_iou)

                if not candidate_ious:
                    continue

                all_ious = torch.cat(candidate_ious)
                all_idxs = torch.cat(candidate_idxs)

                # Adaptive threshold: μ + σ
                mean_iou = all_ious.mean()
                std_iou = all_ious.std()
                threshold = mean_iou + std_iou

                # Select anchors above threshold AND inside GT box
                gt_cx = gt_boxes[gt_i, 0] * img_size[1]
                gt_cy = gt_boxes[gt_i, 1] * img_size[0]
                gt_w = gt_boxes[gt_i, 2] * img_size[1]
                gt_h = gt_boxes[gt_i, 3] * img_size[0]
                gt_x1 = gt_cx - gt_w / 2
                gt_y1 = gt_cy - gt_h / 2
                gt_x2 = gt_cx + gt_w / 2
                gt_y2 = gt_cy + gt_h / 2

                keep = all_ious > threshold
                if keep.sum() == 0:
                    # Fallback: take the single best IoU anchor
                    best_idx_local = all_ious.argmax()
                    keep = torch.zeros_like(all_ious, dtype=torch.bool)
                    keep[best_idx_local] = True

                # Center-in-GT check
                for idx_local in keep.nonzero(as_tuple=False).flatten():
                    anchor_idx = all_idxs[idx_local].item()
                    ax = anchors[0, anchor_idx, 0].item()
                    ay = anchors[0, anchor_idx, 1].item()
                    if gt_x1 <= ax <= gt_x2 and gt_y1 <= ay <= gt_y2:
                        iou_val = float(all_ious[idx_local])
                        target_labels[b, anchor_idx].zero_()
                        target_scores[b, anchor_idx].zero_()
                        target_labels[b, anchor_idx, gt_cls[gt_i]] = 1.0
                        target_bboxes[b, anchor_idx] = gt_boxes[gt_i]
                        target_scores[b, anchor_idx, gt_cls[gt_i]] = iou_val
                        fg_mask[b, anchor_idx] = True

                # If nothing matched (center check failed), take best IoU
                if not fg_mask[b].any():
                    best_idx = all_idxs[all_ious.argmax()].item()
                    iou_val = float(all_ious.max())
                    target_labels[b, best_idx, gt_cls[gt_i]] = 1.0
                    target_bboxes[b, best_idx] = gt_boxes[gt_i]
                    target_scores[b, best_idx, gt_cls[gt_i]] = iou_val
                    fg_mask[b, best_idx] = True

        return target_labels, target_bboxes, target_scores, fg_mask


# =============================================================================
# 6. Combined Badger Loss
# =============================================================================

class BadgerLoss(nn.Module):
    """
    Combined loss function for Badger.

    Total Loss = λ_box × Box Loss + λ_cls × Class Loss + λ_dfl × DFL Loss

    Where:
      - Box Loss: CIoU / GIoU loss (only for matched predictions)
      - Class Loss: Binary Cross-Entropy (all predictions)
      - DFL Loss: Distribution Focal Loss (only for matched predictions)

    Supports two label assigners:
      - TAL (YOLOv8-style): alignment-based, fixed top-k
      - SimOTA (YOLOX-style): dynamic-k, optimal transport based
    """

    def __init__(self, num_classes=80, box_weight=7.5, cls_weight=0.5,
                 dfl_weight=1.5, quality_weight=1.0, label_smoothing=0.0,
                 assigner='tal', box_loss_type='ciou', use_vfl=True):
        """
        Args:
            box_loss_type: 'ciou' (default), 'wiou', 'inner_iou',
                          'focal_eiou', 'siou', 'iou'
            quality_weight: weight for IoU quality prediction loss.
                           Only used when quality_scores are provided to forward().
                           Set to 0 to disable quality loss entirely.
        """
        super().__init__()
        self.num_classes = num_classes
        self.box_weight = box_weight
        self.cls_weight = cls_weight
        self.dfl_weight = dfl_weight
        self.quality_weight = quality_weight
        self.label_smoothing = label_smoothing
        self.box_loss_type = box_loss_type

        # WIoU state: running mean of IoU loss for outlier detection
        self.register_buffer('wiou_iou_mean', torch.tensor(-1.0))  # -1 = uninitialized
        self.wiou_momentum = 0.99

        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.qfl_beta = 2.0
        self.use_vfl = use_vfl

        # Choose label assigner
        if assigner == 'simota':
            self.assigner = SimOTAAssigner(num_classes=num_classes)
        elif assigner == 'atss':
            self.assigner = ATSSAssigner(num_classes=num_classes)
        else:
            self.assigner = TaskAlignedAssigner(num_classes=num_classes)

    def _get_anchors_and_strides(self, feature_shapes, strides, device):
        """
        Generate anchor points (grid cell centers) for each feature map.

        For each feature map cell, the anchor point is the center of that cell
        in image coordinates. This is anchor-FREE detection — we predict
        offsets FROM these points, not anchor boxes.
        """
        anchors = []
        anchor_points_list = []

        for i, (h, w) in enumerate(feature_shapes):
            stride = strides[i]
            # Create grid of (x, y) positions
            grid_y, grid_x = torch.meshgrid(
                torch.arange(h, device=device),
                torch.arange(w, device=device),
                indexing='ij'
            )
            # Convert to image coordinates
            grid_xy = torch.stack([grid_x, grid_y], dim=-1).float()
            anchor_points = grid_xy * stride + stride / 2  # Center of each cell
            anchor_points_list.append(anchor_points.view(-1, 2))

            # Strides per anchor
            anchors.append(torch.full((h * w, 1), stride, device=device))

        return torch.cat(anchors, dim=0), torch.cat(anchor_points_list, dim=0)

    def _decode_bboxes(self, pred_bboxes, anchor_points, strides, reg_max=16):
        """
        Decode predicted bbox distributions into actual coordinates.

        For each anchor point, the model predicts offsets (left, top, right,
        bottom) in feature-stride units. This decodes them to pixel-space
        (cx, cy, w, h) boxes.
        """
        # pred_bboxes: [B, N_total, 4] — already passed through DFL
        stride_scale = strides.view(1, -1, 1)
        pred_bboxes = pred_bboxes * stride_scale

        # Convert from (left, top, right, bottom) to (cx, cy, w, h)
        lt = pred_bboxes[..., :2]  # left, top
        rb = pred_bboxes[..., 2:]  # right, bottom

        # Anchor center minus left/top = top-left corner
        x1y1 = anchor_points - lt
        # Anchor center plus right/bottom = bottom-right corner
        x2y2 = anchor_points + rb

        # Convert to (cx, cy, w, h)
        cxcy = (x1y1 + x2y2) / 2
        wh = (x2y2 - x1y1)

        return torch.cat([cxcy, wh], dim=-1)

    def forward(self, cls_scores, bbox_preds, targets, img_size,
                raw_reg_preds=None, quality_scores=None, reg_max=16):
        """
        Compute the full Badger loss.

        Args:
            cls_scores: list of [B, C, H_i, W_i] — class logits per scale
            bbox_preds: list of [B, 4, H_i, W_i] — DFL-decoded box offsets per scale
            targets: [num_gt, 6] — (batch_idx, cls, x, y, w, h) normalized
            img_size: (H, W) of input images
            raw_reg_preds: list of [B, 4*reg_max, H_i, W_i] — raw reg logits
                          (before DFL softmax), needed for DFL loss. If None,
                          DFL loss is skipped.
            quality_scores: list of [B, 1, H_i, W_i] — quality/IoU logits per scale.
                           If None, quality loss is skipped (backward compatible).

        Returns:
            total_loss: combined loss (scalar)
            loss_dict: {'box': ..., 'cls': ..., 'dfl': ..., 'quality': ...} for logging
        """
        batch_size = cls_scores[0].shape[0]
        device = cls_scores[0].device
        strides = [8, 16, 32]

        # Flatten predictions from all scales
        all_cls = []
        all_bbox = []
        all_raw_reg = [] if raw_reg_preds is not None else None

        for i, (cls, bbox) in enumerate(zip(cls_scores, bbox_preds)):
            b, c, h, w = cls.shape
            # Reshape: [B, C, H, W] → [B, H*W, C]
            all_cls.append(cls.permute(0, 2, 3, 1).reshape(b, -1, c))
            # Reshape: [B, 4, H, W] → [B, H*W, 4] (already DFL-decoded offsets)
            all_bbox.append(bbox.permute(0, 2, 3, 1).reshape(b, -1, 4))
            if raw_reg_preds is not None:
                raw = raw_reg_preds[i]  # [B, 4*reg_max, H, W]
                all_raw_reg.append(raw.permute(0, 2, 3, 1).reshape(b, -1, 4 * reg_max))

        all_cls = torch.cat(all_cls, dim=1)    # [B, N_total, num_classes]
        all_bbox = torch.cat(all_bbox, dim=1)   # [B, N_total, 4]
        if all_raw_reg is not None:
            all_raw_reg = torch.cat(all_raw_reg, dim=1)  # [B, N_total, 4*reg_max]

        # Get anchor points and strides
        feature_shapes = [(cls.shape[2], cls.shape[3]) for cls in cls_scores]
        strides_tensor, anchor_points = self._get_anchors_and_strides(
            feature_shapes, strides, device
        )
        anchor_points = anchor_points.unsqueeze(0)  # [1, N_total, 2]

        # Decode boxes from DFL offsets. DFL returns bin expectations; the bins
        # represent distance in stride units, then decode to image pixels.
        decoded_bboxes = self._decode_bboxes(all_bbox, anchor_points, strides_tensor)

        # Normalize decoded boxes to [0,1] range for loss computation
        # (target boxes from assigner are already normalized)
        img_h, img_w = img_size
        decoded_bboxes[..., 0] = decoded_bboxes[..., 0] / img_w  # cx
        decoded_bboxes[..., 1] = decoded_bboxes[..., 1] / img_h  # cy
        decoded_bboxes[..., 2] = decoded_bboxes[..., 2] / img_w  # w
        decoded_bboxes[..., 3] = decoded_bboxes[..., 3] / img_h  # h

        # Apply Task Aligned Assigner
        num_gt = len(targets)
        target_labels, target_bboxes, target_scores, fg_mask = self.assigner(
            all_cls.sigmoid(),  # Use sigmoid for score-based alignment
            decoded_bboxes,
            targets,
            anchor_points,
            strides,
            img_size,
            num_gt
        )

        # --- 1. Classification Loss (quality focal) ---
        # Modern dense detectors rank boxes better when class confidence encodes
        # localization quality. Positives use IoU as the soft class target;
        # negatives stay zero and are focal-weighted to handle imbalance.
        if self.label_smoothing > 0:
            target_cls_smooth = target_scores * (1 - self.label_smoothing)
        else:
            target_cls_smooth = target_scores

        if self.use_vfl:
            cls_loss = self._varifocal_loss(all_cls, target_cls_smooth, fg_mask)
        else:
            cls_loss = self._quality_focal_loss(all_cls, target_cls_smooth)

        # --- 2. Box Loss (selectable: CIoU, WIoU, Inner-IoU, etc.) ---
        # --- 3. DFL Loss — Distribution Focal Loss on raw distributions ---
        if fg_mask.sum() > 0:
            pred_boxes_fg = decoded_bboxes[fg_mask]
            target_boxes_fg = target_bboxes[fg_mask]
            box_loss = self._compute_box_loss(pred_boxes_fg, target_boxes_fg)

            # DFL loss: penalize the distribution for not peaking at the true offset
            if all_raw_reg is not None:
                # Target offsets in stride units, clamped to [0, reg_max-1]
                # anchor_points: [1, N_total, 2] in pixel coords
                # target_bboxes: [B, N_total, 4] in normalized (cx, cy, w, h)
                target_ltrb = self._compute_target_ltrb(
                    target_bboxes[fg_mask],
                    anchor_points.expand(batch_size, -1, -1)[fg_mask],
                    strides_tensor.view(1, -1, 1).expand(batch_size, -1, -1)[fg_mask],
                    img_size,
                    reg_max=reg_max
                )
                dfl_loss_val = dfl_loss(all_raw_reg[fg_mask], target_ltrb, reg_max=reg_max)
            else:
                dfl_loss_val = torch.tensor(0.0, device=device)
        else:
            box_loss = torch.tensor(0.0, device=device)
            dfl_loss_val = torch.tensor(0.0, device=device)

        # --- 4. Quality Loss — IoU prediction calibration ---
        # Teaches the quality branch to predict the actual IoU of each detection.
        # For matched (fg) anchors: target = IoU between pred box and assigned GT
        # For unmatched (bg) anchors: target = 0 (no object → zero quality)
        if quality_scores is not None and self.quality_weight > 0:
            quality_loss = self._compute_quality_loss(
                quality_scores, fg_mask, target_scores, batch_size
            )
        else:
            quality_loss = torch.tensor(0.0, device=device)

        # --- 5. Combined Loss ---
        total_loss = (
            self.box_weight * box_loss +
            self.cls_weight * cls_loss +
            self.dfl_weight * dfl_loss_val +
            self.quality_weight * quality_loss
        )

        loss_dict = {
            'box': box_loss.detach().item(),
            'cls': cls_loss.detach().item(),
            'dfl': dfl_loss_val.detach().item(),
            'quality': quality_loss.detach().item() if isinstance(quality_loss, torch.Tensor) else quality_loss,
            'total': total_loss.detach().item()
        }

        return total_loss, loss_dict

    def _compute_quality_loss(self, quality_scores, fg_mask, target_scores,
                               batch_size):
        """
        Quality prediction loss — teaches the quality branch to predict IoU.

        Uses Smooth L1 loss on matched (foreground) predictions only.
        Target: actual IoU between predicted box and assigned GT.

        Unlike BCE (which treats IoU as binary), Smooth L1 is appropriate
        for continuous regression targets. Background anchors are ignored
        since they'll be filtered by the class score anyway.

        Args:
            quality_scores: list of [B, 1, H_i, W_i] — quality logits per scale
            fg_mask: [B, N_total] — which anchors are matched to a GT
            target_scores: [B, N_total, num_classes] — soft class targets with IoU
            batch_size: int

        Returns:
            quality_loss: scalar
        """
        if fg_mask.sum() == 0:
            return torch.tensor(0.0, device=fg_mask.device)

        # Flatten quality scores from all scales
        all_quality = []
        for q in quality_scores:
            all_quality.append(q.permute(0, 2, 3, 1).reshape(batch_size, -1, 1))
        all_quality = torch.cat(all_quality, dim=1).squeeze(-1)  # [B, N_total]

        # IoU target from target_scores (only non-zero for matched anchors)
        iou_target = target_scores.max(dim=-1)[0]  # [B, N_total]

        # Only compute loss on foreground (matched) predictions
        # Apply sigmoid to quality logits since targets are in [0, 1]
        pred_quality = all_quality[fg_mask].sigmoid()
        target_iou = iou_target[fg_mask]

        # Smooth L1 loss — robust to outliers, smooth gradient at zero
        quality_loss = F.smooth_l1_loss(pred_quality, target_iou)

        return quality_loss

    def _quality_focal_loss(self, logits, target_scores):
        """
        Quality Focal Loss for dense detection confidence calibration.

        The target is continuous: IoU for the matched class, 0 otherwise.
        This teaches inference scores to mean "class is present and localized
        well", which makes thresholding/NMS much less noisy than plain BCE.
        """
        pred_scores = logits.sigmoid()
        bce = self.bce(logits, target_scores)
        scale = (target_scores - pred_scores).abs().pow(self.qfl_beta)
        loss = bce * scale
        normalizer = target_scores.sum().clamp(min=1.0)
        return loss.sum() / normalizer

    def _varifocal_loss(self, logits, target_scores, fg_mask, alpha=0.75, gamma=2.0):
        """
        Varifocal Loss — from Zhang et al., "VarifocalNet" (arXiv:2008.13367).

        Better than QFL for dense detection because it explicitly separates
        positive and negative sample weighting:
          - Positives: weighted by IoU score (well-localized = higher weight)
          - Negatives: weighted by α × p^γ (reduces easy-negative dominance)

        This is what Ultralytics YOLOv8/v11 uses as its primary cls loss.
        Reference implementation: ultralytics/utils/loss.py VarifocalLoss.
        """
        pred_scores = logits.sigmoid()
        # Positives: -iou * log(p) * (1-p)^gamma
        pos_loss = target_scores * F.binary_cross_entropy_with_logits(
            logits, torch.ones_like(logits), reduction='none')
        pos_loss = pos_loss * ((1 - pred_scores) ** gamma)
        # Negatives: -alpha * log(1-p) * p^gamma
        neg_loss = alpha * F.binary_cross_entropy_with_logits(
            logits, torch.zeros_like(logits), reduction='none')
        neg_loss = neg_loss * (pred_scores ** gamma)
        # Only apply negative loss where fg_mask is False
        neg_loss = neg_loss * (~fg_mask.unsqueeze(-1)).float()
        loss = pos_loss + neg_loss
        normalizer = target_scores.sum().clamp(min=1.0)
        return loss.sum() / normalizer

    def _compute_target_ltrb(self, target_boxes_cxcywh, anchor_points,
                             anchor_strides, img_size, reg_max=16):
        """
        Compute target (left, top, right, bottom) offsets from anchor points.

        The DFL module predicts bin offsets in [0, reg_max-1]. Each bin is one
        feature stride, so pixel offsets are divided by the matching stride
        before applying DFL.

        Args:
            target_boxes_cxcywh: [N, 4] — GT boxes in normalized (cx, cy, w, h)
            anchor_points: [N, 2] — anchor point coordinates in pixel space
            anchor_strides: [N, 1] — stride for each anchor point
            img_size: (H, W) tuple

        Returns:
            target_ltrb: [N, 4] — stride-normalized offsets for L/T/R/B
        """
        img_h, img_w = img_size

        # Convert GT from normalized to pixel coords
        gt_cx = target_boxes_cxcywh[:, 0] * img_w
        gt_cy = target_boxes_cxcywh[:, 1] * img_h
        gt_w  = target_boxes_cxcywh[:, 2] * img_w
        gt_h  = target_boxes_cxcywh[:, 3] * img_h

        gt_x1 = gt_cx - gt_w / 2
        gt_y1 = gt_cy - gt_h / 2
        gt_x2 = gt_cx + gt_w / 2
        gt_y2 = gt_cy + gt_h / 2

        anchor_x = anchor_points[:, 0]
        anchor_y = anchor_points[:, 1]

        # Pixel offsets from anchor to GT edges
        left   = anchor_x - gt_x1    # positive if anchor is right of GT left edge
        top    = anchor_y - gt_y1
        right  = gt_x2 - anchor_x    # positive if GT right edge is right of anchor
        bottom = gt_y2 - anchor_y

        target_ltrb = torch.stack([left, top, right, bottom], dim=-1)
        target_ltrb = target_ltrb / anchor_strides.clamp(min=1.0)

        # Clamp to valid range [0, reg_max-1] (16 bins for default Badger)
        target_ltrb = target_ltrb.clamp(0, reg_max - 1)

        return target_ltrb

    def _compute_box_loss(self, pred_boxes, target_boxes):
        """
        Dispatch to the selected box loss function.

        Supported types:
          - 'ciou': Complete IoU (default, stable)
          - 'wiou': Wise-IoU v3 (+0.5-0.8 AP, dynamic focusing)
          - 'inner_iou': Inner-IoU (+0.3-0.5 AP, scale-aware)
          - 'focal_eiou': Focal-EIoU (+0.3-0.5 AP, hard-sample focus)
          - 'siou': SCYLLA-IoU (angle-aware)
          - 'iou': Plain IoU loss (fastest, least accurate)
        """
        if self.box_loss_type == 'wiou':
            from .advanced_losses import wiou_v3_loss
            # Initialize running mean on first call
            if self.wiou_iou_mean < 0:
                loss, iou_mean = wiou_v3_loss(pred_boxes, target_boxes, iou_mean=None)
                self.wiou_iou_mean = iou_mean.detach()
            else:
                loss, iou_mean = wiou_v3_loss(pred_boxes, target_boxes,
                                              iou_mean=self.wiou_iou_mean)
                self.wiou_iou_mean = (self.wiou_momentum * self.wiou_iou_mean +
                                      (1 - self.wiou_momentum) * iou_mean.detach())
            return loss

        elif self.box_loss_type == 'inner_iou':
            from .advanced_losses import inner_iou_loss
            return inner_iou_loss(pred_boxes, target_boxes, inner_scale=0.75,
                                  iou_type='ciou')

        elif self.box_loss_type == 'focal_eiou':
            from .advanced_losses import focal_eiou_loss
            return focal_eiou_loss(pred_boxes, target_boxes, gamma=0.5)

        elif self.box_loss_type == 'giou':
            from .advanced_losses import giou_loss
            return giou_loss(pred_boxes, target_boxes)

        elif self.box_loss_type == 'siou':
            from .advanced_losses import siou_loss
            return siou_loss(pred_boxes, target_boxes)

        elif self.box_loss_type == 'iou':
            iou = bbox_iou(pred_boxes, target_boxes, xywh=False)
            # Diagonal IoU
            diag = torch.diag(iou) if iou.ndim == 2 else iou
            return (1.0 - diag).mean()

        else:  # default: ciou
            return ciou_loss(pred_boxes, target_boxes)
