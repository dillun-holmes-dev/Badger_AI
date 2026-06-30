"""
Loss functions package.
"""
from .badger_loss import BadgerLoss, ciou_loss, dfl_loss, TaskAlignedAssigner, SimOTAAssigner
from .advanced_losses import GaussianDFL, VarifocalLoss, siou_loss
