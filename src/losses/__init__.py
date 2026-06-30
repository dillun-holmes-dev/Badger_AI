"""
🦡 Badger AI — Unified Loss Functions.

All losses in one clean package. No v1/v2 split.
"""
from .badger_loss import BadgerLoss, ciou_loss, dfl_loss, TaskAlignedAssigner, SimOTAAssigner
from .advanced_losses import (GaussianDFL, VarifocalLoss, siou_loss,
                               wiou_v3_loss, inner_iou_loss,
                               focal_eiou_loss, compute_box_loss)
