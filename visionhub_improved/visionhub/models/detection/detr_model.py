"""Detection-only DETR wrapper."""

from __future__ import annotations

from torch import nn


class DETRDetection(nn.Module):
    def __init__(self, backbone, encoder, transformer):
        super().__init__()
        self.backbone = backbone
        self.encoder = encoder
        self.transformer = transformer

    def deploy(self):
        self.eval()
        for module in self.modules():
            if hasattr(module, "convert_to_deploy"):
                module.convert_to_deploy()
        return self

    def forward(self, samples, targets=None):
        x = samples.tensors if hasattr(samples, "tensors") else samples
        feats = self.backbone(x)
        feats = self.encoder(feats)
        return self.transformer(feats, targets, x if self.training else None)
