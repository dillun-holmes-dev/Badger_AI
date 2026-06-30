"""
Knowledge Distillation for Badger.

Train a small "student" model to mimic a large "teacher" model.
This is one of the highest-ROI techniques for improving small models —
you get most of the teacher's accuracy at a fraction of the cost.

The math:
  L_total = L_task + λ_KD × L_KD

Where:
  - L_task: standard detection loss (CIoU + BCE + DFL) on ground truth
  - L_KD: distillation loss — student predictions should match teacher's

Three distillation strategies (use one or combine):
  1. Feature distillation: student's intermediate features match teacher's
  2. Logit distillation: student's raw outputs match teacher's (soft labels)
  3. Detection distillation: student's final boxes/scores match teacher's

Reference papers:
  - Hinton et al., "Distilling the Knowledge in a Neural Network" (2015)
    arXiv:1503.02531 — the original distillation paper
  - Chen et al., "Learning Efficient Object Detection Models with KD" (2017)
    arXiv:1705.02451 — detection-specific distillation
  - Yang et al., "FGD: Focal and Global Knowledge Distillation for Detectors"
    (CVPR 2022) — state-of-the-art detection distillation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy


# =============================================================================
# 1. Feature Distillation
# =============================================================================

class FeatureDistiller(nn.Module):
    """
    Distill intermediate feature maps from teacher to student.

    For each feature scale (P3, P4, P5):
      L_feat = MSE(adapt(student_feat), teacher_feat)

    The adaptation layer projects student features to match teacher's
    channel count, since student typically has fewer channels.

    This is the most effective distillation strategy for detection
    because intermediate features encode rich spatial information.
    """

    def __init__(self, student_channels, teacher_channels):
        """
        Args:
            student_channels: list of channels per scale [c3, c4, c5]
            teacher_channels: list of channels per scale [c3, c4, c5]
        """
        super().__init__()
        self.adaptors = nn.ModuleList()
        for s_ch, t_ch in zip(student_channels, teacher_channels):
            self.adaptors.append(
                nn.Conv2d(s_ch, t_ch, 1, bias=False)
            )

    def forward(self, student_feats, teacher_feats):
        """
        Args:
            student_feats: list of [B, C_s, H, W] from student
            teacher_feats: list of [B, C_t, H, W] from teacher (detached)

        Returns:
            feature distillation loss (scalar)
        """
        loss = 0.0
        for s_feat, t_feat, adapt in zip(student_feats, teacher_feats, self.adaptors):
            s_adapted = adapt(s_feat)
            # MSE between adapted student and teacher features
            loss += F.mse_loss(s_adapted, t_feat.detach())
        return loss / len(student_feats)


# =============================================================================
# 2. Logit Distillation (Soft Labels)
# =============================================================================

def distillation_loss(student_logits, teacher_logits, temperature=3.0):
    """
    KL divergence between softened student and teacher predictions.

    Higher temperature → softer probability distributions → more
    information about class relationships (e.g., "cat is more similar
    to dog than to car" is captured in the soft targets).

    Math:
      L_KD = T² × KL(softmax(teacher/T) || softmax(student/T))

    The T² factor compensates for the gradient scaling effect of
    temperature on softmax.

    Args:
        student_logits: [B, C, H, W] — raw logits from student
        teacher_logits: [B, C, H, W] — raw logits from teacher (detached)
        temperature: softening factor (>1 = softer, more informative)

    Returns:
        distillation loss (scalar)
    """
    # Soften predictions
    student_soft = F.log_softmax(student_logits / temperature, dim=1)
    teacher_soft = F.softmax(teacher_logits.detach() / temperature, dim=1)

    # KL divergence
    kd_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean')

    # Scale by T²
    return kd_loss * (temperature ** 2)


# =============================================================================
# 3. Detection Distillation (Box + Score matching)
# =============================================================================

def detection_distillation_loss(student_cls, student_bbox,
                                teacher_cls, teacher_bbox,
                                fg_mask, cls_weight=1.0, box_weight=1.0):
    """
    Distill final detection outputs: classification scores + bounding boxes.

    Only distills on foreground predictions (where teacher has high confidence
    and there's actual signal to transfer).

    Args:
        student_cls: [B, N, num_classes] — student class logits
        student_bbox: [B, N, 4] — student box predictions
        teacher_cls: [B, N, num_classes] — teacher class logits (detached)
        teacher_bbox: [B, N, 4] — teacher box predictions (detached)
        fg_mask: [B, N] — foreground mask (which predictions to distill)
        cls_weight, box_weight: loss component weights

    Returns:
        dict with 'cls_kd' and 'box_kd' losses
    """
    if fg_mask.sum() == 0:
        return {'cls_kd': torch.tensor(0.0), 'box_kd': torch.tensor(0.0)}

    # Classification distillation: MSE on sigmoid outputs (softer than logits)
    s_cls_fg = student_cls[fg_mask].sigmoid()
    t_cls_fg = teacher_cls[fg_mask].sigmoid().detach()
    cls_loss = F.mse_loss(s_cls_fg, t_cls_fg)

    # Box distillation: smooth L1 on box coordinates
    s_box_fg = student_bbox[fg_mask]
    t_box_fg = teacher_bbox[fg_mask].detach()
    box_loss = F.smooth_l1_loss(s_box_fg, t_box_fg)

    return {
        'cls_kd': cls_weight * cls_loss,
        'box_kd': box_weight * box_loss,
    }


# =============================================================================
# 4. Combined Distillation Trainer
# =============================================================================

class DistillationTrainer:
    """
    Complete distillation training loop.

    Usage:
        teacher = create_model('badger-x', num_classes=80)
        teacher.load_state_dict(torch.load('badger-x-coco.pth'))

        student = create_model('badger-s', num_classes=80)
        distiller = DistillationTrainer(teacher, student)

        for epoch in range(epochs):
            for batch in dataloader:
                total_loss, loss_dict = distiller.training_step(batch)
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
    """

    def __init__(self, teacher, student,
                 feature_weight=1.0,
                 logit_weight=0.5,
                 detection_weight=1.0,
                 task_weight=1.0,
                 temperature=3.0):
        """
        Args:
            teacher: pretrained large model (frozen)
            student: small model to train
            feature_weight: weight for feature distillation loss
            logit_weight: weight for logit/soft distillation loss
            detection_weight: weight for detection output distillation
            task_weight: weight for standard detection loss (ground truth)
            temperature: softening temperature for logit distillation
        """
        self.teacher = teacher
        self.student = student
        self.feature_weight = feature_weight
        self.logit_weight = logit_weight
        self.detection_weight = detection_weight
        self.task_weight = task_weight
        self.temperature = temperature

        # Freeze teacher
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.teacher.eval()

        # Feature adaptors (if student/teacher channel counts differ)
        s_channels = self.student.backbone.out_channels
        t_channels = self.teacher.backbone.out_channels
        self.feature_distiller = FeatureDistiller(s_channels, t_channels)

    @torch.no_grad()
    def _teacher_forward(self, images):
        """Run teacher inference and return features + predictions."""
        features = self.teacher.backbone(images)
        fused = self.teacher.neck(features)
        cls_scores, bbox_preds = self.teacher.head(fused)
        return features, cls_scores, bbox_preds

    def training_step(self, images, targets=None, loss_fn=None):
        """
        Full distillation training step.

        Args:
            images: [B, 3, H, W] input batch
            targets: [N, 6] ground truth (can be None for unsupervised KD)
            loss_fn: BadgerLoss instance for task loss (optional)

        Returns:
            total_loss, loss_dict
        """
        B = images.shape[0]

        # 1. Teacher forward (frozen, no grad)
        with torch.no_grad():
            t_features, t_cls, t_bbox = self._teacher_forward(images)

        # 2. Student forward (with grad)
        s_features = self.student.backbone(images)
        s_fused = self.student.neck(s_features)
        s_cls, s_bbox = self.student.head(s_fused)

        losses = {}

        # 3. Feature distillation loss
        if self.feature_weight > 0:
            feat_loss = self.feature_distiller(s_features, t_features)
            losses['kd_feat'] = self.feature_weight * feat_loss

        # 4. Logit distillation loss (per scale)
        if self.logit_weight > 0:
            logit_loss = 0.0
            for s_c, t_c in zip(s_cls, t_cls):
                logit_loss += distillation_loss(s_c, t_c, self.temperature)
            losses['kd_logit'] = self.logit_weight * logit_loss

        # 5. Detection distillation (requires foreground mask)
        if self.detection_weight > 0 and loss_fn is not None and targets is not None:
            # Flatten predictions
            all_s_cls = torch.cat([c.permute(0, 2, 3, 1).reshape(B, -1, c.shape[1]) for c in s_cls], dim=1)
            all_t_cls = torch.cat([c.permute(0, 2, 3, 1).reshape(B, -1, c.shape[1]) for c in t_cls], dim=1)
            all_s_bbox = torch.cat([b.permute(0, 2, 3, 1).reshape(B, -1, 4) for b in s_bbox], dim=1)
            all_t_bbox = torch.cat([b.permute(0, 2, 3, 1).reshape(B, -1, 4) for b in t_bbox], dim=1)

            # Use teacher's confidence as foreground mask
            t_conf = all_t_cls.sigmoid().max(dim=-1)[0]  # [B, N]
            fg_mask = t_conf > 0.5

            det_losses = detection_distillation_loss(
                all_s_cls, all_s_bbox, all_t_cls, all_t_bbox, fg_mask
            )
            losses['kd_cls'] = self.detection_weight * det_losses['cls_kd']
            losses['kd_box'] = self.detection_weight * det_losses['box_kd']

        # 6. Task loss (standard detection loss on ground truth)
        if self.task_weight > 0 and loss_fn is not None and targets is not None:
            task_loss, task_dict = loss_fn(s_cls, s_bbox, targets, images.shape[-2:])
            losses['task'] = self.task_weight * task_loss
            for k, v in task_dict.items():
                losses[f'task_{k}'] = v

        total_loss = sum(losses.values())
        return total_loss, losses
