from visionhub.core import LazyCall as L
from visionhub.models.detection import DenseDetectionPostProcess, RTMODetectionCriterion
from visionhub.models.rtmo import RTMODetection
from visionhub.nn import HGNetv2

eval_spatial_size = (640, 640)
feat_strides = [8, 16, 32]
num_classes = 2
internal_num_body_points = 0

training_params = {
    "clip_max_norm": 0.1,
    "save_checkpoint_interval": 1,
    "grad_accum_steps": 1,
    "print_freq": 100,
    "sync_bn": False,
    "use_ema": False,
    "dist_url": "env://",
}

postprocessor = L(DenseDetectionPostProcess)(
    score_threshold=0.3,
    nms_threshold=0.65,
    max_detections=300,
    deploy_mode=False,
)

model = L(RTMODetection)(
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
    num_body_points=internal_num_body_points,
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

criterion = L(RTMODetectionCriterion)(
    num_classes=num_classes,
    weight_dict={"loss_cls": 2.0, "loss_bbox": 5.0, "loss_bbox_aux": 1.0},
    topk_candidates=10,
    center_radius=2.5,
    assign_cls_weight=1.0,
    assign_iou_weight=3.0,
    overlaps_power=0.5,
    use_aux_loss=True,
)
