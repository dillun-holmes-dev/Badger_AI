"""
RTMOPose base model/criterion/postprocessor definitions.
Imported by size-specific config files.
"""

from visionhub.core import LazyCall as L
from visionhub.models.rtmo import RTMOPose, RTMOCriterion, RTMOPostProcess
from visionhub.nn import HGNetv2
import numpy as np

# Shared parameters (overridden by dataset-specific configs)
eval_spatial_size = (640, 640)
feat_strides = [8, 16, 32]
num_classes = 2
num_body_points = 17

training_params = {
    "clip_max_norm": 0.1,
    "save_checkpoint_interval": 1,
    "grad_accum_steps": 1,
    "print_freq": 100,
    "sync_bn": False,
    "use_ema": False,
    "dist_url": "env://",
}

postprocessor = L(RTMOPostProcess)(
    score_threshold=0.3,
    nms_threshold=0.45,
    max_detections=300,
    bbox_padding=1.25,
    deploy_mode=False,
)

model = L(RTMOPose)(
    backbone=L(HGNetv2)(
        name="B0",
        use_lab=True,
        return_idx=[1, 2, 3],
        freeze_stem_only=True,
        freeze_at=-1,
        freeze_norm=True,
        pretrained=True,
    ),
    neck_out_channels=256,
    neck_depth_mult=0.34,
    num_classes=num_classes,
    num_body_points=num_body_points,
    feat_strides=feat_strides,
    head_num_convs=3,
    head_num_groups=8,
    head_channels_per_group=36,
    dcc_feat_channels=128,
    dcc_num_bins=(192, 256),
    dcc_spe_channels=128,
    bbox_padding=1.25,
    act="silu",
    post_processor=postprocessor,
)

criterion = L(RTMOCriterion)(
    num_classes=num_classes,
    num_body_points=num_body_points,
    sigmas=None,
    weight_dict={
        "loss_cls":  3.0,
        "loss_bbox": 5.0,
        "loss_kpts": 20.0,
        "loss_vis":  1.0,
        "loss_mle":  1.0,
        "loss_bbox_aux": 1.0,
    },
    topk_candidates=10,
    center_radius=2.5,
    assign_cls_weight=2.0,
    assign_iou_weight=3.0,
    assign_oks_weight=1.5,
    assign_vis_weight=0.0,
    dynamic_k_indicator="hybrid",
    use_keypoints_for_center=True,
    overlaps_power=1.0,
    bbox_padding=1.25,
    use_aux_loss=True,
    use_dcc_loss=True,
)
