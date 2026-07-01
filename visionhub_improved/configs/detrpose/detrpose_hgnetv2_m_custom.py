from .include.detrpose_hgnetv2 import model, criterion, training_params, postprocessor
from .include.dataset_custom import (
    dataset_train, dataset_val, dataset_test, evaluator,
    NUM_CLASSES, NUM_BODY_POINTS, CLASS_MAPPINGS, CLASS_SKELETONS, SIGMAS
)

from visionhub.core import LazyCall as L
from visionhub.nn.optimizer import ModelEMA
from visionhub.misc.get_param_dicts import get_optim_params

from torch import optim

# ── Output / training ────────────────────────────────────────────────────────
training_params["output_dir"] = "output/detrpose_hgnetv2_m_custom"
training_params["epochs"] = 100
training_params["use_ema"] = True
training_params["grad_accum_steps"] = 1

ema = L(ModelEMA)(
    decay=0.9999,
    warmups=2000
    )

# ── Optimizer ────────────────────────────────────────────────────────────────
optimizer = L(optim.AdamW)(
    params=L(get_optim_params)(
        cfg=[
                {
                'params': '^(?=.*backbone).*$',
                'lr': 0.00001
                },
            ],
        ),
    lr=0.0001,
    betas=[0.9, 0.999],
    weight_decay=0.0001
    )

lr_scheduler = L(optim.lr_scheduler.MultiStepLR)(
    milestones=[80],
    gamma=0.1
    )

# ── Backbone (HGNetV2-M / B2) ─────────────────────────────────────────────
model.backbone.name = 'B2'
model.backbone.use_lab = True
model.encoder.in_channels = [384, 768, 1536]
model.encoder.depth_mult = 0.67
model.transformer.num_decoder_layers = 4

# ── Dataset-derived parameters (read from data/coco/train/coco_instances.json)
model.transformer.num_body_points = NUM_BODY_POINTS
model.transformer.num_classes = NUM_CLASSES
model.transformer.sigmas = SIGMAS

criterion.num_classes = NUM_CLASSES
criterion.num_body_points = NUM_BODY_POINTS
criterion.sigmas = SIGMAS
criterion.matcher.num_body_points = NUM_BODY_POINTS
criterion.matcher.cost_class = 2.0
criterion.matcher.cost_keypoints = 10.0
criterion.matcher.cost_oks = 4.0

postprocessor.num_body_points = NUM_BODY_POINTS

