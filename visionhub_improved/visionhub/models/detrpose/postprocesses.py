"""
DETRPose: Real-time end-to-end transformer model for multi-person pose estimation
Copyright (c) 2025 The DETRPose Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR/)
Copyright (c) 2023 RT-DETR Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from GroupPose (https://github.com/Michel-liu/GroupPose/)
Copyright (c) 2023 GroupPose Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from ED-Pose (https://github.com/IDEA-Research/ED-Pose/)
Copyright (c) 2023 IDEA. All Rights Reserved.
"""

import torch
from torch import nn
from torchvision.ops.boxes import nms

from ...misc.box_ops import box_cxcywh_to_xyxy


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    def __init__(self, num_select=60, num_body_points=17, nms_iou_threshold=0.65, use_nms=True) -> None:
        super().__init__()
        self.num_select = num_select
        self.num_body_points = num_body_points
        self.nms_iou_threshold = nms_iou_threshold
        self.use_nms = use_nms
        self.deploy_mode = False

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        out_logits, out_keypoints= outputs['pred_logits'], outputs['pred_keypoints']
        out_boxes = outputs['pred_boxes']  # [bs, nq, 4] normalized cxcywh
        batch_size, _, num_classes = out_logits.shape
        num_select = min(self.num_select, out_logits.shape[1] * num_classes)

        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), num_select, dim=1)
        scores = topk_values

        # query indices for gathering
        topk_keypoints = torch.div(topk_indexes, num_classes, rounding_mode='floor')
        labels = topk_indexes % num_classes
        
        if self.deploy_mode:
            img_h, img_w = target_sizes.unbind(1)

            if out_keypoints.dim() == 4:
                keypoints = torch.gather(
                    out_keypoints,
                    1,
                    topk_keypoints[..., None, None].expand(batch_size, num_select, self.num_body_points, 2),
                )
                keypoint_scale = torch.stack([img_w, img_h], dim=1)
                keypoints = keypoints * keypoint_scale[:, None, None, :]
            else:
                keypoints = torch.gather(
                    out_keypoints,
                    1,
                    topk_keypoints.unsqueeze(-1).expand(batch_size, num_select, self.num_body_points * 2),
                )
                keypoint_scale = torch.stack([img_w, img_h], dim=1).repeat(1, self.num_body_points)
                keypoints = keypoints * keypoint_scale[:, None, :]
                keypoints = keypoints.unflatten(-1, (self.num_body_points, 2))

            boxes = torch.gather(out_boxes, 1, topk_keypoints.unsqueeze(-1).expand(batch_size, num_select, 4))
            scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)[:, None, :]
            boxes = box_cxcywh_to_xyxy(boxes) * scale_fct
            return scores, labels, keypoints, boxes

        keypoints = torch.gather(out_keypoints, 1, topk_keypoints.unsqueeze(-1).repeat(1, 1, self.num_body_points*2))
        img_h, img_w = target_sizes.unbind(1)
        keypoint_scale = torch.stack([img_w, img_h], dim=1).repeat(1, self.num_body_points)
        keypoints = keypoints * keypoint_scale[:, None, :]
        keypoints_res = keypoints.unflatten(-1, (-1, 2))
        keypoints_res = torch.cat(
            [keypoints_res, torch.ones_like(keypoints_res[..., 0:1])], 
            dim=-1).flatten(-2)

        # gather and scale bounding boxes
        boxes = torch.gather(out_boxes, 1, topk_keypoints.unsqueeze(-1).repeat(1, 1, 4))
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)[:, None, :]
        boxes = box_cxcywh_to_xyxy(boxes) * scale_fct

        results = []
        for s, l, k, b in zip(scores, labels, keypoints_res, boxes):
            if self.use_nms and b.shape[0] > 0:
                # Class-aware NMS: offset boxes by class id so different classes don't suppress each other
                max_coord = b.max()
                offsets = l.float() * (max_coord + 1)
                boxes_for_nms = b + offsets[:, None]
                keep = nms(boxes_for_nms, s, self.nms_iou_threshold)
                s, l, k, b = s[keep], l[keep], k[keep], b[keep]
            results.append({'scores': s, 'labels': l, 'keypoints': k, 'boxes': b})
        return results

    def deploy(self, ):
        self.eval()
        self.deploy_mode = True
        return self
