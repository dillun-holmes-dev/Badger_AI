"""
Loss functions package.

v1: Standard losses (CIoU, BCE, DFL)
v2: State-of-the-art losses (WIoU v3, Inner-IoU, Focal-EIoU)
"""
# --- v1 Losses ---
from .badger_loss import BadgerLoss, ciou_loss, dfl_loss, TaskAlignedAssigner, SimOTAAssigner
from .advanced_losses import GaussianDFL, VarifocalLoss, siou_loss

# --- v2 Losses (state-of-the-art) ---
from .sota_losses import wiou_loss, inner_iou_loss, focal_eiou_loss, compute_box_loss

