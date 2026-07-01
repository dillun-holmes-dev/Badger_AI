"""
MMPose-style SimOTA assignment for standalone RTMO.

This mirrors the important behavior of mmpose.models.task_modules.assigners:
  - priors are [x, y, stride_x, stride_y]
  - valid candidates are inside a GT box or its center region
  - final pair candidates must be inside both the box and center region
  - cost combines BCE class cost, -log(IoU), -log(OKS), optional visibility BCE
  - dynamic-k can be driven by IoU or OKS
"""

import torch
import torch.nn.functional as F

from visionhub.misc.box_ops import box_iou


INF = 100000.0
EPS = 1.0e-7


def _oks_overlaps(pred_keypoints, gt_keypoints, gt_visible, gt_areas, sigmas, eps=1e-8):
    """Pairwise OKS, matching MMPose PoseOKS semantics."""
    if pred_keypoints is None or gt_keypoints is None or gt_visible is None or gt_areas is None:
        return None

    sigmas = sigmas.to(device=pred_keypoints.device, dtype=pred_keypoints.dtype)
    gt_visible = gt_visible.to(dtype=pred_keypoints.dtype)
    dist = torch.norm(pred_keypoints[:, None] - gt_keypoints[None], dim=-1)
    dist = dist / gt_areas[None, :, None].pow(0.5).clamp(min=eps)
    dist = dist / (sigmas.view(1, 1, -1) * 2.0).clamp(min=eps)

    weights = gt_visible / gt_visible.sum(dim=-1, keepdim=True).clamp(min=eps)
    return (torch.exp(-dist.pow(2) / 2.0) * weights[None]).sum(dim=-1)


def _get_gt_centers(gt_boxes, gt_keypoints, gt_visible, use_keypoints_for_center):
    centers = (gt_boxes[:, :2] + gt_boxes[:, 2:]) / 2.0
    if not use_keypoints_for_center or gt_keypoints is None or gt_visible is None:
        return centers

    visible = gt_visible > 0
    visible_count = visible.sum(dim=-1, keepdim=True)
    has_visible = visible_count.squeeze(-1) > 0
    if has_visible.any():
        kpt_centers = (gt_keypoints * visible.unsqueeze(-1)).sum(dim=-2)
        kpt_centers = kpt_centers / visible_count.clamp(min=1).to(gt_keypoints.dtype)
        centers = centers.clone()
        centers[has_visible] = kpt_centers[has_visible].to(centers)
    return centers


def _get_in_gt_and_center_info(
    priors,
    gt_boxes,
    gt_keypoints=None,
    gt_visible=None,
    center_radius=2.5,
    use_keypoints_for_center=False,
):
    num_gt = gt_boxes.size(0)
    repeated_x = priors[:, 0].unsqueeze(1).repeat(1, num_gt)
    repeated_y = priors[:, 1].unsqueeze(1).repeat(1, num_gt)
    repeated_stride_x = priors[:, 2].unsqueeze(1).repeat(1, num_gt)
    repeated_stride_y = priors[:, 3].unsqueeze(1).repeat(1, num_gt)

    l_ = repeated_x - gt_boxes[:, 0]
    t_ = repeated_y - gt_boxes[:, 1]
    r_ = gt_boxes[:, 2] - repeated_x
    b_ = gt_boxes[:, 3] - repeated_y
    is_in_gts = torch.stack([l_, t_, r_, b_], dim=1).min(dim=1).values > 0
    is_in_gts_all = is_in_gts.sum(dim=1) > 0

    gt_centers = _get_gt_centers(gt_boxes, gt_keypoints, gt_visible, use_keypoints_for_center)
    gt_cxs, gt_cys = gt_centers[:, 0], gt_centers[:, 1]

    ct_box_l = gt_cxs - center_radius * repeated_stride_x
    ct_box_t = gt_cys - center_radius * repeated_stride_y
    ct_box_r = gt_cxs + center_radius * repeated_stride_x
    ct_box_b = gt_cys + center_radius * repeated_stride_y

    cl_ = repeated_x - ct_box_l
    ct_ = repeated_y - ct_box_t
    cr_ = ct_box_r - repeated_x
    cb_ = ct_box_b - repeated_y
    is_in_cts = torch.stack([cl_, ct_, cr_, cb_], dim=1).min(dim=1).values > 0
    is_in_cts_all = is_in_cts.sum(dim=1) > 0

    valid_mask = is_in_gts_all | is_in_cts_all
    is_in_boxes_and_centers = is_in_gts[valid_mask] & is_in_cts[valid_mask]
    return valid_mask, is_in_boxes_and_centers


def _dynamic_k_matching(cost, pairwise_scores, num_gt, valid_mask, topk_candidates):
    matching_matrix = torch.zeros_like(cost, dtype=torch.bool)
    candidate_topk = min(int(topk_candidates), pairwise_scores.size(0))
    if candidate_topk <= 0:
        full_fg = torch.zeros_like(valid_mask)
        empty = torch.zeros(0, dtype=torch.long, device=cost.device)
        empty_score = cost.new_zeros((0,))
        return full_fg, empty, empty_score

    topk_scores, _ = torch.topk(pairwise_scores, candidate_topk, dim=0)
    dynamic_ks = torch.clamp(topk_scores.sum(0).int(), min=1)

    # Vectorized: find top-candidate_topk lowest-cost anchors for all GTs at once,
    # then mask each GT's column to its dynamic_k entries — avoids .item() syncs.
    _, topk_cost_indices = torch.topk(cost, candidate_topk, dim=0, largest=False)  # [K, G]
    ranks = torch.arange(candidate_topk, device=cost.device).unsqueeze(1)          # [K, 1]
    k_mask = ranks < dynamic_ks.unsqueeze(0)                                        # [K, G]
    matching_matrix.scatter_(0, topk_cost_indices, k_mask)

    multi_match = matching_matrix.sum(1) > 1
    if multi_match.any():
        _, cost_argmin = torch.min(cost[multi_match], dim=1)
        matching_matrix[multi_match] = False
        matching_matrix[multi_match, cost_argmin] = True

    fg_valid = matching_matrix.sum(1) > 0
    valid_indices = valid_mask.nonzero(as_tuple=False).squeeze(1)
    full_fg = torch.zeros_like(valid_mask)
    full_fg[valid_indices[fg_valid]] = True

    matched_gt_inds = matching_matrix[fg_valid].to(torch.long).argmax(1)
    matched_scores = (matching_matrix.to(pairwise_scores.dtype) * pairwise_scores).sum(1)[fg_valid]
    return full_fg, matched_gt_inds, matched_scores


@torch.no_grad()
def simota_assign(
    cls_scores,
    bbox_xyxy,
    priors,
    gt_boxes,
    gt_labels,
    gt_keypoints=None,
    gt_keypoints_visible=None,
    pred_keypoints=None,
    pred_keypoints_visible=None,
    gt_areas=None,
    sigmas=None,
    topk_candidates=10,
    center_radius=2.5,
    lambda_cls=1.0,
    lambda_iou=3.0,
    lambda_oks=3.0,
    lambda_vis=0.0,
    dynamic_k_indicator="oks",
    use_keypoints_for_center=True,
):
    """
    Assign ground-truth targets to RTMO priors.

    Args:
        cls_scores: [N, C] sigmoid probabilities, usually sqrt(cls_prob).
        bbox_xyxy: [N, 4] decoded predicted boxes.
        priors: [N, 4] point priors as [x, y, stride_x, stride_y].
    """
    N = cls_scores.shape[0]
    G = gt_boxes.shape[0]
    device = cls_scores.device

    if G == 0 or N == 0:
        fg_mask = torch.zeros(N, dtype=torch.bool, device=device)
        dummy = torch.zeros(0, dtype=torch.long, device=device)
        return fg_mask, dummy, dummy, gt_boxes[:0], cls_scores.new_zeros((0,))

    valid_mask, is_in_boxes_and_centers = _get_in_gt_and_center_info(
        priors,
        gt_boxes,
        gt_keypoints=gt_keypoints,
        gt_visible=gt_keypoints_visible,
        center_radius=center_radius,
        use_keypoints_for_center=use_keypoints_for_center,
    )

    if valid_mask.sum() == 0:
        fg_mask = torch.zeros(N, dtype=torch.bool, device=device)
        dummy = torch.zeros(0, dtype=torch.long, device=device)
        return fg_mask, dummy, dummy, gt_boxes[:0], cls_scores.new_zeros((0,))

    valid_boxes = bbox_xyxy[valid_mask]
    valid_scores = cls_scores[valid_mask].clamp(min=EPS, max=1.0 - EPS)
    num_valid = valid_boxes.size(0)

    cost = (~is_in_boxes_and_centers).to(valid_boxes.dtype) * INF

    pairwise_ious, _ = box_iou(valid_boxes, gt_boxes)
    if lambda_iou > 0:
        cost = cost + (-torch.log(pairwise_ious + EPS)) * lambda_iou

    pairwise_oks = None
    needs_oks_scores = dynamic_k_indicator in {"oks", "hybrid"}
    if (lambda_oks > 0 or needs_oks_scores) and pred_keypoints is not None:
        pairwise_oks = _oks_overlaps(
            pred_keypoints[valid_mask],
            gt_keypoints,
            gt_keypoints_visible,
            gt_areas,
            sigmas,
        )
        if pairwise_oks is not None and lambda_oks > 0:
            cost = cost + (-torch.log(pairwise_oks + EPS)) * lambda_oks

    if lambda_cls > 0:
        gt_onehot = F.one_hot(gt_labels.to(torch.int64), cls_scores.shape[-1]).float()
        gt_onehot = gt_onehot.unsqueeze(0).repeat(num_valid, 1, 1)
        valid_scores_expanded = valid_scores.unsqueeze(1).repeat(1, G, 1)
        cls_cost = F.binary_cross_entropy(valid_scores_expanded.float(), gt_onehot, reduction="none").sum(-1)
        cost = cost + cls_cost.to(cost.dtype) * lambda_cls

    if lambda_vis > 0 and pred_keypoints_visible is not None and gt_keypoints_visible is not None:
        pred_vis = pred_keypoints_visible[valid_mask].clamp(min=EPS, max=1.0 - EPS)
        pred_vis = pred_vis.unsqueeze(1).repeat(1, G, 1)
        gt_vis = gt_keypoints_visible.unsqueeze(0).float().repeat(num_valid, 1, 1)
        vis_cost = F.binary_cross_entropy(pred_vis.float(), gt_vis, reduction="none").sum(-1)
        cost = cost + vis_cost.to(cost.dtype) * lambda_vis

    if dynamic_k_indicator == "hybrid" and pairwise_oks is not None:
        total_weight = max(float(lambda_iou + lambda_oks), EPS)
        pairwise_scores = (
            pairwise_ious * float(lambda_iou) + pairwise_oks * float(lambda_oks)
        ) / total_weight
    elif dynamic_k_indicator == "oks" and pairwise_oks is not None:
        pairwise_scores = pairwise_oks
    else:
        pairwise_scores = pairwise_ious

    fg_mask, assigned_gt, assigned_score = _dynamic_k_matching(
        cost, pairwise_scores, G, valid_mask, topk_candidates
    )

    assigned_cls = gt_labels[assigned_gt]
    assigned_box = gt_boxes[assigned_gt]
    return fg_mask, assigned_gt, assigned_cls, assigned_box, assigned_score
