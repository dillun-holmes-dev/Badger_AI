"""
Badger Training Utilities — advanced training at SOTA level.

Components:
  - trainer.py:   Advanced training loop (EMA, SWA, AMP, grad accumulation)
  - evolution.py: Hyperparameter evolution via genetic algorithm
  - augment.py:   Mosaic9, MixUp, Copy-Paste augmentations
"""
from .trainer import BadgerTrainer, create_optimizer, create_scheduler
from .evolution import HyperparameterEvolution, EvolutionConfig

try:
    from .augment import MosaicAugment, MixUpAugment, CopyPasteAugment, AugmentationPipeline
except ImportError:
    # cv2 may not be installed — augmentations will be unavailable
    MosaicAugment = None
    MixUpAugment = None
    CopyPasteAugment = None
    AugmentationPipeline = None
