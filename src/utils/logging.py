"""
Logging utilities for training.

Supports:
  - TensorBoard logging
  - Console progress bars with tqdm
  - Checkpoint saving/loading
"""

import os
import torch
from datetime import datetime
from pathlib import Path


class Logger:
    """
    Unified logger for training that writes to both console and TensorBoard.

    Usage:
        logger = Logger(log_dir='runs/exp1', use_tensorboard=True)
        logger.log_scalar('loss', 1.23, step=100)
        logger.log_scalars({'loss_box': 1.0, 'loss_cls': 0.5}, step=100)
    """

    def __init__(self, log_dir='runs/', use_tensorboard=True, use_wandb=False):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.use_tensorboard = use_tensorboard
        self.use_wandb = use_wandb
        self.writer = None

        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(log_dir=str(self.log_dir))
            except ImportError:
                print("Warning: tensorboard not installed. Skipping.")
                self.use_tensorboard = False

        if use_wandb:
            try:
                import wandb
                self.wandb = wandb
            except ImportError:
                print("Warning: wandb not installed. Skipping.")
                self.use_wandb = False

        self.metrics_history = {}

    def log_scalar(self, tag, value, step):
        """Log a single scalar value."""
        if self.use_tensorboard and self.writer:
            self.writer.add_scalar(tag, value, step)

        if tag not in self.metrics_history:
            self.metrics_history[tag] = []
        self.metrics_history[tag].append((step, value))

    def log_scalars(self, tag_dict, step):
        """Log multiple scalars at once."""
        for tag, value in tag_dict.items():
            self.log_scalar(tag, value, step)

    def log_text(self, tag, text, step):
        """Log text (e.g., config summary)."""
        if self.use_tensorboard and self.writer:
            self.writer.add_text(tag, text, step)

    def close(self):
        """Close the logger."""
        if self.writer:
            self.writer.close()
        if self.use_wandb:
            self.wandb.finish()


def save_checkpoint(model, optimizer, epoch, loss, path, config=None):
    """
    Save a training checkpoint.

    Args:
        model: the model (state_dict saved)
        optimizer: the optimizer (state_dict saved)
        epoch: current epoch number
        loss: current loss value
        path: file path to save to
        config: optional config dict to include
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'config': config,
    }
    torch.save(checkpoint, path)


def load_checkpoint(path, model, optimizer=None):
    """
    Load a training checkpoint.

    Args:
        path: checkpoint file path
        model: model to load weights into
        optimizer: optional optimizer to load state into

    Returns:
        epoch, loss, config
    """
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return (
        checkpoint.get('epoch', 0),
        checkpoint.get('loss', float('inf')),
        checkpoint.get('config', None)
    )


def setup_experiment_dir(base_dir='runs', experiment_name=None):
    """
    Create a timestamped experiment directory.

    Returns:
        Path to the experiment directory
    """
    if experiment_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        experiment_name = f'badger_{timestamp}'

    exp_dir = Path(base_dir) / experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir
