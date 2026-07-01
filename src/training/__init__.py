"""
Badger Training Utilities — advanced training at SOTA level.

Components:
  - trainer.py:   Advanced training loop (EMA, SWA, AMP, grad accumulation)
  - supermind.py: Auto-LR, one-cycle, "it just works" trainer
  - evolution.py: Hyperparameter evolution via genetic algorithm
  - augment.py:   Mosaic9, MixUp, Copy-Paste augmentations
"""
from .trainer import BadgerTrainer, create_optimizer, create_scheduler, ModelEMA, SWA
from .supermind import SuperMind, LRFinder, OneCycleScheduler, easy_train
from .evolution import HyperparameterEvolution, EvolutionConfig

try:
    from .augment import MosaicAugment, MixUpAugment, CopyPasteAugment, AugmentationPipeline
except ImportError:
    MosaicAugment = None
    MixUpAugment = None
    CopyPasteAugment = None
    AugmentationPipeline = None
