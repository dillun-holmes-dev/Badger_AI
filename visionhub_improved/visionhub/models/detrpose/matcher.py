"""
DETRPose: Real-time end-to-end transformer model for multi-person pose estimation
Copyright (c) 2025 The DETRPose Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from GroupPose (https://github.com/Michel-liu/GroupPose/)
Copyright (c) 2023 GroupPose Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from ED-Pose (https://github.com/IDEA-Research/ED-Pose/)
Copyright (c) 2023 IDEA. All Rights Reserved.
"""

import torch
from torch import nn
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor

from visionhub.misc.keypoint_loss import resolve_sigmas

try:
    import lapjv
except ModuleNotFoundError:
    lapjv = None
    try:
        from scipy.optimize import linear_sum_assignment as scipy_linear_sum_assignment
    except ModuleNotFoundError:
        scipy_linear_sum_assignment = None
else:
    scipy_linear_sum_assignment = None

# Module-level thread pool for parallel Hungarian matching across batch items.
# lapjv is a C extension that releases the GIL, so threads run in true parallel.
_MATCHER_POOL = ThreadPoolExecutor(max_workers=min(os.cpu_count() or 4, 16))


def _lapjv_solve(cost_np):
    """Solve assignment using lapjv when available, otherwise scipy."""
    n_rows, n_cols = cost_np.shape
    if n_rows == 0 or n_cols == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    if lapjv is None:
        if scipy_linear_sum_assignment is None:
            raise ModuleNotFoundError(
                "HungarianMatcher requires either `lapjv` or `scipy`. "
                "Install one of them to use DETRPose matching."
            )
        row_ind, col_ind = scipy_linear_sum_assignment(cost_np.astype(np.float64))
        return row_ind.astype(np.int64), col_ind.astype(np.int64)

    # lapjv requires a square float64 matrix; pad rectangular matrices with a
    # large sentinel so dummy assignments are strongly avoided.
    size = max(n_rows, n_cols)
    if n_rows != n_cols:
        padded = np.full((size, size), 1e9, dtype=np.float64)
        padded[:n_rows, :n_cols] = cost_np.astype(np.float64)
    else:
        padded = cost_np.astype(np.float64)
    _, x, _ = lapjv.lapjv(padded)           # x[row] = assigned_col
    row_ind = np.where(x[:n_rows] < n_cols)[0]
    col_ind = x[row_ind]
    return row_ind, col_ind


class HungarianMatcher(nn.Module):
    def __init__(self, cost_class: float = 1, focal_alpha=0.25,
                 cost_keypoints=1.0, cost_oks=0.01, num_body_points=17, sigmas=None):
        super().__init__()
        self.cost_class = cost_class

        self.cost_keypoints = cost_keypoints
        self.cost_oks = cost_oks
        self.focal_alpha = focal_alpha
        self.num_body_points = num_body_points
        self.set_sigmas(sigmas)

    def set_sigmas(self, sigmas):
        self.sigmas = resolve_sigmas(sigmas, self.num_body_points)
        return self

    @torch.no_grad()
    def forward(self, outputs, targets):
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()  # [batch_size * num_queries, num_classes]
        out_keypoints = outputs["pred_keypoints"].flatten(0, 1)  # [batch_size * num_queries, 51]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        sizes = [len(v["boxes"]) for v in targets]
        if tgt_ids.numel() == 0:
            return [
                (
                    torch.empty(0, dtype=torch.int64),
                    torch.empty(0, dtype=torch.int64),
                )
                for _ in targets
            ]

        tgt_keypoints = torch.cat([v["keypoints"] for v in targets])  # nkp, 51
        tgt_area = torch.cat([v["area"] for v in targets])  # nkp, 51

        # Compute the classification cost.
        alpha = self.focal_alpha
        gamma = 2.0
        neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
        pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
        cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

        # compute the keypoint costs
        Z_pred = out_keypoints[:, 0:(self.num_body_points * 2)]
        Z_gt = tgt_keypoints[:, 0:(self.num_body_points * 2)]
        V_gt: torch.Tensor = tgt_keypoints[:, (self.num_body_points * 2):]
        sigmas = Z_pred.new_tensor(self.sigmas)
        variances = (sigmas * 2) ** 2
        kpt_preds = Z_pred.reshape(-1, Z_pred.size(-1) // 2, 2)
        kpt_gts = Z_gt.reshape(-1, Z_gt.size(-1) // 2, 2)
        squared_distance = (kpt_preds[:, None, :, 0] - kpt_gts[None, :, :, 0]) ** 2 + \
                           (kpt_preds[:, None, :, 1] - kpt_gts[None, :, :, 1]) ** 2
        squared_distance0 = squared_distance / (tgt_area[:, None].clamp(min=1e-6) * variances[None, :] * 2)
        squared_distance1 = torch.exp(-squared_distance0) * V_gt
        visible_count = V_gt.sum(dim=-1).clamp(min=1e-6)
        oks = (squared_distance1.sum(dim=-1) / visible_count).clamp(min=1e-6)
        cost_oks = 1 - oks

        cost_keypoints = torch.abs(Z_pred[:, None, :] - Z_gt[None])  # npred, ngt, 34
        cost_keypoints = cost_keypoints * V_gt.repeat_interleave(2, dim=1)[None]
        cost_keypoints = cost_keypoints.sum(-1)
        C = self.cost_class * cost_class + self.cost_keypoints * cost_keypoints + self.cost_oks * cost_oks
        C = C.view(bs, num_queries, -1).detach().cpu()

        # Final cost matrix - move to CPU once, then solve all batch items in parallel
        cost_splits = C.split(sizes, -1)
        futures = [_MATCHER_POOL.submit(_lapjv_solve, cost_splits[i][i].numpy())
                   for i in range(len(sizes))]
        indices = [f.result() for f in futures]

        if tgt_ids.shape[0] > 0:
            cost_mean_dict = {
                'class': cost_class.mean(),
                "keypoints": cost_keypoints.mean()
            }
        else:
            # for the cases when no grounding truth boxes
            cost_mean_dict = {
                'class': torch.zeros_like(cost_class.mean()),
                'keypoints': torch.zeros_like(cost_keypoints.mean()),
            }

        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in
                indices]#, cost_mean_dict

def build_matcher(args):
    assert args.matcher_type in ['HungarianMatcher'], "Unknown args.matcher_type: {}".format(
        args.matcher_type)
    if args.matcher_type == 'HungarianMatcher':
        return HungarianMatcher(
            cost_class=args.set_cost_class,
            focal_alpha=args.focal_alpha,
            cost_keypoints=args.set_cost_keypoints,
            cost_oks=args.set_cost_oks,
            num_body_points=args.num_body_points,
            sigmas=getattr(args, "sigmas", None),
        )
    else:
        raise NotImplementedError("Unknown args.matcher_type: {}".format(args.matcher_type))
