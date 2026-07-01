"""
RTMDet-Pose-N (nano) – HGNetV2-B0, 128-ch neck, depth 0.34.
Smallest / fastest configuration.
"""

from .include.rtmdetpose_hgnetv2 import model, criterion, training_params, postprocessor
from .include.dataset_custom import (
    dataset_train, dataset_val, dataset_test, evaluator,
    NUM_CLASSES, NUM_BODY_POINTS, CLASS_MAPPINGS, CLASS_SKELETONS, SIGMAS,
)

from visionhub.core import LazyCall as L
from visionhub.nn.optimizer import ModelEMA
from visionhub.misc.get_param_dicts import get_optim_params
from torch import optim

# ── Output / training ─────────────────────────────────────────────────────────
training_params["output_dir"] = "output/rtmdetpose_hgnetv2_n_custom"
training_params["epochs"] = 100
training_params["use_ema"] = True
training_params["grad_accum_steps"] = 1

ema = L(ModelEMA)(decay=0.9999, warmups=2000)

# ── Optimizer ─────────────────────────────────────────────────────────────────
optimizer = L(optim.AdamW)(
    params=L(get_optim_params)(
        cfg=[{"params": "^(?=.*backbone).*$", "lr": 0.0001}],
    ),
    lr=0.0001,
    betas=[0.9, 0.999],
    weight_decay=0.0001,
)

lr_scheduler = L(optim.lr_scheduler.MultiStepLR)(
    milestones=[80], gamma=0.1
)

# ── Backbone / neck ───────────────────────────────────────────────────────────
model.backbone.name = "B0"
model.backbone.use_lab = True
model.neck_depth_mult = 0.34
model.neck_out_channels = 128

# ── Dataset-derived parameters ────────────────────────────────────────────────
model.num_classes     = NUM_CLASSES
model.num_body_points = NUM_BODY_POINTS

criterion.num_classes     = NUM_CLASSES
criterion.num_body_points = NUM_BODY_POINTS
criterion.sigmas          = SIGMAS

postprocessor.score_threshold = 0.3
