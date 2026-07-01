import torch
import numpy as np
import torch.nn as nn


def resolve_sigmas(sigmas, num_keypoints, dtype=np.float32):
    num_keypoints = int(num_keypoints)
    if num_keypoints < 0:
        raise ValueError("num_keypoints must be non-negative.")

    if sigmas is None:
        values = np.ones(num_keypoints, dtype=dtype) * 0.05
    elif isinstance(sigmas, torch.Tensor):
        values = sigmas.detach().cpu().numpy().astype(dtype, copy=False)
    else:
        values = np.asarray(sigmas, dtype=dtype)

    if values.ndim != 1:
        raise ValueError("sigmas must be a 1D sequence.")
    if values.shape[0] != num_keypoints:
        raise ValueError(
            f"Expected {num_keypoints} sigma values, got {values.shape[0]}."
        )
    return np.ascontiguousarray(values, dtype=dtype)

def oks_overlaps(kpt_preds, kpt_gts, kpt_valids, kpt_areas, sigmas):
    sigmas = kpt_preds.new_tensor(sigmas)
    variances = (sigmas * 2)**2

    assert kpt_preds.size(0) == kpt_gts.size(0)
    kpt_preds = kpt_preds.reshape(-1, kpt_preds.size(-1) // 2, 2)
    kpt_gts = kpt_gts.reshape(-1, kpt_gts.size(-1) // 2, 2)

    squared_distance = (kpt_preds[:, :, 0] - kpt_gts[:, :, 0]) ** 2 + \
        (kpt_preds[:, :, 1] - kpt_gts[:, :, 1]) ** 2
    squared_distance0 = squared_distance / (kpt_areas[:, None] * variances[None, :] * 2)
    squared_distance1 = torch.exp(-squared_distance0)
    squared_distance1 = squared_distance1 * kpt_valids
    oks = squared_distance1.sum(dim=1) / (kpt_valids.sum(dim=1)+1e-6)

    return oks

def oks_loss(pred,
             target,
             valid=None,
             area=None,
             linear=False,
             sigmas=None,
             eps=1e-6):
    oks = oks_overlaps(pred, target, valid, area, sigmas).clamp(min=eps)
    if linear:
        loss = oks
    else:
        loss = -oks.log()
    return loss


class OKSLoss(nn.Module):
    def __init__(self,
                 linear=False,
                 num_keypoints=17,
                 sigmas=None,
                 eps=1e-6,
                 reduction='mean',
                 loss_weight=1.0):
        super(OKSLoss, self).__init__()
        self.linear = linear
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.sigmas = resolve_sigmas(sigmas, num_keypoints)

    def forward(self,
                pred,
                target,
                valid,
                area,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if (weight is not None) and (not torch.any(weight > 0)) and (
                reduction != 'none'):
            if pred.dim() == weight.dim() + 1:
                weight = weight.unsqueeze(1)
            return (pred * weight).sum()  # 0
        if weight is not None and weight.dim() > 1:
            # TODO: remove this in the future
            # reduce the weight of shape (n, 4) to (n,) to match the
            # iou_loss of shape (n,)
            assert weight.shape == pred.shape
            weight = weight.mean(-1)
        loss = self.loss_weight * oks_loss(
            pred,
            target,
            valid=valid,
            area=area,
            linear=self.linear,
            sigmas=self.sigmas,
            eps=self.eps)
        return loss
