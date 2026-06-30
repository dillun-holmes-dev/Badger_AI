"""
Advanced loss functions for improving detection accuracy.

Key mathematical improvements over standard YOLO losses:

1. Gaussian DFL (Distribution Focal Loss):
   Instead of discrete bins, models the edge distribution as a
   Gaussian mixture, giving continuous-valued predictions with
   uncertainty estimates. This is how DEIMv2/D-FINE get their
   final ~0.3 AP boost.

2. Varifocal Loss:
   Extends Focal Loss by weighting positive samples by their IoU
   score — predictions that are well-localized contribute more
   to the classification loss. This couples localization quality
   with classification confidence.

3. SIoU Loss:
   Adds angle cost to the standard CIoU — penalizes boxes with
   mismatched orientations, especially helpful for rotated objects.

Reference papers:
  - Li et al., "Generalized Focal Loss" (arXiv:2006.04388)
     → Introduces DFL for bounding box distributions
  - Zhang et al., "VarifocalNet" (arXiv:2008.13367)
     → IoU-aware classification loss  
  - Gevorgyan, "SIoU Loss" (arXiv:2205.12740)
     → Angle-aware box regression loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# =============================================================================
# Gaussian Distribution Focal Loss
# =============================================================================

class GaussianDFL(nn.Module):
    """
    Gaussian Distribution Focal Loss — continuous version of DFL.

    Standard DFL (in badger_loss.py):
      - Predicts discrete probabilities over reg_max bins per edge
      - Loss: cross-entropy between predicted distribution and target
      - Limitation: discretization error — the true edge offset may fall
        between bins

    Gaussian DFL:
      - Predicts (μ, σ) per edge — mean and std of a Gaussian distribution
      - Loss: negative log-likelihood of target under predicted Gaussian
      - Advantage: continuous predictions, no discretization error,
        AND we get uncertainty estimates from σ

    The predicted distribution for edge offset e is:
      P(e) = (1 / (σ√(2π))) × exp(-(e - μ)² / (2σ²))

    Loss: L = -log P(e_target)
         = log(σ) + (e_target - μ)² / (2σ²) + log(√(2π))
    """

    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred_mu, pred_sigma, target):
        """
        Args:
            pred_mu:    [N, 4] — predicted mean for each edge (l, t, r, b)
            pred_sigma: [N, 4] — predicted std for each edge (must be > 0)
            target:     [N, 4] — ground truth edge offsets

        Returns:
            loss: scalar — negative log-likelihood
        """
        # Ensure sigma is positive (softplus)
        sigma = F.softplus(pred_sigma) + 1e-6

        # Gaussian negative log-likelihood:
        # -log P(x|μ,σ) = log(σ) + (x-μ)²/(2σ²) + ½log(2π)
        squared_error = (target - pred_mu) ** 2
        loss = torch.log(sigma) + squared_error / (2 * sigma ** 2) + 0.5 * math.log(2 * math.pi)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


# =============================================================================
# Varifocal Loss — IoU-aware classification
# =============================================================================

class VarifocalLoss(nn.Module):
    """
    Varifocal Loss: Focal Loss where positive sample weight = IoU score.

    Standard BCE/Focal: all positives weighted equally
    Varifocal:          high-IoU positives weighted MORE

    This elegantly couples classification and localization:
      - A box that's perfectly placed (high IoU) → high classification weight
      - A box that's poorly placed (low IoU) → low classification weight

    The math:
      VFL(p, q) = -q × (q × log(p) + (1-q) × log(1-p))

    where q is the target score: IoU for positives, 0 for negatives.
    This means the model learns to predict higher confidence for better-
    localized boxes — exactly what we want.

    Reference: Zhang et al., "VarifocalNet", arXiv:2008.13367
    """

    def __init__(self, alpha=0.75, gamma=2.0, reduction='mean'):
        """
        Args:
            alpha: balance factor for negative samples
            gamma: focusing parameter (like Focal Loss)
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target, iou_scores=None):
        """
        Args:
            pred:       [N, C] — predicted logits
            target:     [N, C] — binary target labels
            iou_scores: [N, C] — IoU scores for positive samples (0 for negatives)

        Returns:
            loss: scalar
        """
        pred_sigmoid = pred.sigmoid()

        if iou_scores is None:
            iou_scores = target  # Fall back to standard BCE

        # For positive samples (target=1): weight = iou_score
        # Loss = -iou * log(p) * (1-p)^γ
        pos_loss = iou_scores * torch.log(pred_sigmoid.clamp(1e-7))
        pos_loss = pos_loss * ((1 - pred_sigmoid) ** self.gamma)

        # For negative samples (target=0): weight = α
        # Loss = -α * log(1-p) * p^γ
        neg_weight = self.alpha * ((1 - target) ** 4)  # Reduce weight for "background"
        neg_loss = neg_weight * torch.log((1 - pred_sigmoid).clamp(1e-7))
        neg_loss = neg_loss * (pred_sigmoid ** self.gamma)

        # Combine
        loss = -(pos_loss + neg_loss)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


# =============================================================================
# SIoU Loss — adds angle awareness to CIoU
# =============================================================================

def siou_loss(pred_boxes, target_boxes, eps=1e-7):
    """
    SIoU (SCYLLA-IoU) Loss — extends CIoU with angle cost.

    Components:
      1. Angle cost — penalizes boxes with misaligned orientations
      2. Distance cost — center distance (like CIoU)
      3. Shape cost — aspect ratio (like CIoU)
      4. IoU cost — overlap

    The angle cost is the novel part:
      Λ = 1 - 2 × sin²(arcsin(x) - π/4)
      where x = |Δy| / √(Δx² + Δy²)
    This penalizes boxes where the center offset is at a 45° angle
    most heavily, decreasing for horizontal/vertical alignments.

    For rotated objects (aerial imagery, text detection), this is
    significantly better than CIoU.

    Reference: Gevorgyan, "SIoU Loss", arXiv:2205.12740
    """
    # Convert (cx, cy, w, h) → (x1, y1, x2, y2)
    p_x1 = pred_boxes[..., 0] - pred_boxes[..., 2] / 2
    p_y1 = pred_boxes[..., 1] - pred_boxes[..., 3] / 2
    p_x2 = pred_boxes[..., 0] + pred_boxes[..., 2] / 2
    p_y2 = pred_boxes[..., 1] + pred_boxes[..., 3] / 2

    t_x1 = target_boxes[..., 0] - target_boxes[..., 2] / 2
    t_y1 = target_boxes[..., 1] - target_boxes[..., 3] / 2
    t_x2 = target_boxes[..., 0] + target_boxes[..., 2] / 2
    t_y2 = target_boxes[..., 1] + target_boxes[..., 3] / 2

    # Box dimensions
    p_w = p_x2 - p_x1
    p_h = p_y2 - p_y1
    t_w = t_x2 - t_x1
    t_h = t_y2 - t_y1

    # Center coordinates
    p_cx = (p_x1 + p_x2) / 2
    p_cy = (p_y1 + p_y2) / 2
    t_cx = (t_x1 + t_x2) / 2
    t_cy = (t_y1 + t_y2) / 2

    # --- Angle cost ---
    delta_x = p_cx - t_cx
    delta_y = p_cy - t_cy

    # sin(α) = opposite / hypotenuse for the angle between centers
    sigma = (delta_x ** 2 + delta_y ** 2).sqrt() + eps
    sin_alpha = (delta_y.abs() / sigma).clamp(0, 1)

    # Λ = sin(2α) = 2 × sin(α) × cos(α)
    #   = 2 × sin(α) × √(1 - sin²(α))
    # Special case: Λ = 1 - 2 × sin²(arcsin(sin_alpha) - π/4)
    # Which simplifies to: Λ = sin(2α)
    sin_2alpha = 2 * sin_alpha * (1 - sin_alpha ** 2).sqrt()
    # Angle cost: 1 - sin²(2α) → penalizes non-horizontal/vertical alignments
    angle_cost = 1 - 2 * sin_2alpha ** 2

    # --- Distance cost ---
    # Uses angle to determine if Δx or Δy dominates
    rho_x = (delta_x / sigma) ** 2
    rho_y = (delta_y / sigma) ** 2
    gamma = 2 - angle_cost  # More penalty when angle is large
    distance_cost = (1 - (-gamma * rho_x).exp()) + (1 - (-gamma * rho_y).exp())

    # --- Shape cost ---
    # Ω = Σ (1 - e^(-|Δsize|))^θ
    # Penalizes aspect ratio differences
    theta = 4.0
    omega_w = (p_w - t_w).abs() / (torch.max(p_w, t_w) + eps)
    omega_h = (p_h - t_h).abs() / (torch.max(p_h, t_h) + eps)
    shape_cost = ((1 - (-omega_w).exp()) ** theta) + ((1 - (-omega_h).exp()) ** theta)

    # --- IoU cost ---
    inter_x1 = torch.max(p_x1, t_x1)
    inter_y1 = torch.max(p_y1, t_y1)
    inter_x2 = torch.min(p_x2, t_x2)
    inter_y2 = torch.min(p_y2, t_y2)
    inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)

    p_area = p_w * p_h
    t_area = t_w * t_h
    union_area = p_area + t_area - inter_area + eps
    iou = inter_area / union_area

    # --- Combined SIoU loss ---
    siou = 1 - iou + (distance_cost + shape_cost) / 2
    return siou.mean()
