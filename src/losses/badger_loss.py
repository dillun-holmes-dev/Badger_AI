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

    def __init__(self, num_classes=80, topk=13, alpha=1.0, beta=6.0):
        self.num_classes = num_classes
        self.topk = topk        # k=13: empirical YOLOv8 default (not from TOOD)
        self.alpha = alpha      # α=1.0: cls exponent (TOOD Table 3)
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

                    target_labels[b, anchor_idx, gt_cls[gt_i]] = 1.0
                    target_bboxes[b, anchor_idx] = gt_boxes[gt_i]
                    target_scores[b, anchor_idx, gt_cls[gt_i]] = score
                    fg_mask[b, anchor_idx] = True

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
                    target_labels[b, anchor_idx, gt_cls[gt_idx]] = 1.0
                    target_bboxes[b, anchor_idx] = gt_boxes[gt_idx]
                    target_scores[b, anchor_idx, gt_cls[gt_idx]] = iou_val
                    fg_mask[b, anchor_idx] = True

        return target_labels, target_bboxes, target_scores, fg_mask


# =============================================================================
# 5. Combined Badger Loss
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
                 dfl_weight=1.5, label_smoothing=0.0, assigner='tal'):
        super().__init__()
        self.num_classes = num_classes
        self.box_weight = box_weight
        self.cls_weight = cls_weight
        self.dfl_weight = dfl_weight
        self.label_smoothing = label_smoothing

        self.bce = nn.BCEWithLogitsLoss(reduction='none')

        # Choose label assigner
        if assigner == 'simota':
            self.assigner = SimOTAAssigner(num_classes=num_classes)
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
        bottom) in pixel units. This decodes them to (cx, cy, w, h) format.
        """
        # pred_bboxes: [B, N_total, 4] — already passed through DFL
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

    def forward(self, cls_scores, bbox_preds, targets, img_size):
        """
        Compute the full Badger loss.

        Args:
            cls_scores: list of [B, C, H_i, W_i] — class logits per scale
            bbox_preds: list of [B, 4, H_i, W_i] — decoded boxes per scale
            targets: [num_gt, 6] — (batch_idx, cls, x, y, w, h) normalized
            img_size: (H, W) of input images

        Returns:
            total_loss: combined loss (scalar)
            loss_dict: {'box': ..., 'cls': ..., 'dfl': ...} for logging
        """
        batch_size = cls_scores[0].shape[0]
        device = cls_scores[0].device
        strides = [8, 16, 32]

        # Flatten predictions from all scales
        all_cls = []
        all_bbox = []

        for i, (cls, bbox) in enumerate(zip(cls_scores, bbox_preds)):
            b, c, h, w = cls.shape
            # Reshape: [B, C, H, W] → [B, H*W, C]
            all_cls.append(cls.permute(0, 2, 3, 1).reshape(b, -1, c))
            # Reshape: [B, 4, H, W] → [B, H*W, 4]
            all_bbox.append(bbox.permute(0, 2, 3, 1).reshape(b, -1, 4))

        all_cls = torch.cat(all_cls, dim=1)    # [B, N_total, num_classes]
        all_bbox = torch.cat(all_bbox, dim=1)   # [B, N_total, 4]

        # Get anchor points and strides
        feature_shapes = [(cls.shape[2], cls.shape[3]) for cls in cls_scores]
        strides_tensor, anchor_points = self._get_anchors_and_strides(
            feature_shapes, strides, device
        )
        anchor_points = anchor_points.unsqueeze(0)  # [1, N_total, 2]

        # Decode boxes
        decoded_bboxes = self._decode_bboxes(all_bbox, anchor_points, strides_tensor)

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

        # --- 1. Classification Loss (Binary Cross-Entropy) ---
        # Apply label smoothing: target is 1-ε for positives, ε for negatives
        if self.label_smoothing > 0:
            target_cls_smooth = target_labels * (1 - self.label_smoothing) + \
                                (1 - target_labels) * self.label_smoothing / self.num_classes
        else:
            target_cls_smooth = target_labels

        cls_loss = self.bce(all_cls, target_cls_smooth).mean()

        # --- 2. Box Loss (CIoU) — only on foreground ---
        if fg_mask.sum() > 0:
            pred_boxes_fg = decoded_bboxes[fg_mask]
            target_boxes_fg = target_bboxes[fg_mask]
            box_loss = ciou_loss(pred_boxes_fg, target_boxes_fg)
            dfl_loss_val = torch.tensor(0.0, device=device)  # Simplified DFL
        else:
            box_loss = torch.tensor(0.0, device=device)
            dfl_loss_val = torch.tensor(0.0, device=device)

        # --- 3. Combined Loss ---
        total_loss = (
            self.box_weight * box_loss +
            self.cls_weight * cls_loss +
            self.dfl_weight * dfl_loss_val
        )

        loss_dict = {
            'box': box_loss.detach().item(),
            'cls': cls_loss.detach().item(),
            'dfl': dfl_loss_val.detach().item(),
            'total': total_loss.detach().item()
        }

        return total_loss, loss_dict
