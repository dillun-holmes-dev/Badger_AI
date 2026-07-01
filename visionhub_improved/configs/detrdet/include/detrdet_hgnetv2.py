from visionhub.core import LazyCall as L
from visionhub.models.detrpose import Criterion, HybridEncoder, Transformer
from visionhub.models.detection import DETRDetection, DETRDetectionPostProcess, HungarianBBoxMatcher
from visionhub.nn import HGNetv2

training_params = {
    "clip_max_norm": 0.1,
    "save_checkpoint_interval": 1,
    "grad_accum_steps": 1,
    "print_freq": 100,
    "sync_bn": False,
    "use_ema": False,
    "dist_url": "env://",
}

eval_spatial_size = (640, 640)
hidden_dim = 256
n_levels = 3
feat_strides = [8, 16, 32]
num_classes = 2
internal_num_body_points = 1

model = L(DETRDetection)(
    backbone=L(HGNetv2)(
        name="B4",
        use_lab=False,
        return_idx=[1, 2, 3],
        freeze_stem_only=True,
        freeze_at=-1,
        freeze_norm=True,
        pretrained=True,
    ),
    encoder=L(HybridEncoder)(
        in_channels=[512, 1024, 2048],
        feat_strides=feat_strides,
        n_levels=n_levels,
        hidden_dim=hidden_dim,
        nhead=8,
        dim_feedforward=1024,
        dropout=0.0,
        enc_act="gelu",
        expansion=1.0,
        depth_mult=1.0,
        act="silu",
        temperatureH=20,
        temperatureW=20,
        eval_spatial_size=eval_spatial_size,
    ),
    transformer=L(Transformer)(
        hidden_dim=hidden_dim,
        dropout=0.0,
        nhead=8,
        num_queries=300,
        dim_feedforward=1024,
        num_decoder_layers=6,
        normalize_before=False,
        return_intermediate_dec=True,
        activation="relu",
        num_feature_levels=3,
        dec_n_points=4,
        learnable_tgt_init=True,
        two_stage_type="standard",
        num_body_points=internal_num_body_points,
        aux_loss=True,
        num_classes=num_classes,
        dec_pred_class_embed_share=False,
        dec_pred_pose_embed_share=False,
        two_stage_class_embed_share=False,
        two_stage_bbox_embed_share=False,
        cls_no_bias=False,
        feat_strides=[8, 16, 32],
        eval_spatial_size=eval_spatial_size,
        reg_max=32,
        reg_scale=4,
        dn_number=20,
        dn_label_noise_ratio=0.5,
    ),
)

criterion = L(Criterion)(
    num_classes=num_classes,
    matcher=L(HungarianBBoxMatcher)(
        cost_class=2.0,
        cost_bbox=5.0,
        cost_giou=2.0,
        focal_alpha=0.25,
    ),
    weight_dict={"loss_ce": 2.0, "loss_bbox": 5.0, "loss_giou": 2.0},
    focal_alpha=0.25,
    losses=["labels", "boxes"],
    num_body_points=internal_num_body_points,
)

postprocessor = L(DETRDetectionPostProcess)(num_select=300, nms_iou_threshold=0.65, use_nms=True)
