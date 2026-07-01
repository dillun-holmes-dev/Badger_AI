from .include.rtmodet_hgnetv2 import model, criterion, postprocessor, training_params
from configs.detection.include.dataset_custom import (
    CLASS_MAPPINGS,
    CLASS_SKELETONS,
    NUM_BODY_POINTS,
    NUM_CLASSES,
    dataset_test,
    dataset_train,
    dataset_val,
    evaluator,
)
from torch import optim

from visionhub.core import LazyCall as L
from visionhub.misc.get_param_dicts import get_optim_params
from visionhub.nn.optimizer import ModelEMA

DETECTION_ONLY = True

training_params["output_dir"] = "output/rtmodet_hgnetv2_m_custom"
training_params["epochs"] = 120
training_params["use_ema"] = True
training_params["grad_accum_steps"] = 1
ema = L(ModelEMA)(decay=0.9999, warmups=2000)
optimizer = L(optim.AdamW)(
    params=L(get_optim_params)(cfg=[{"params": "^(?=.*backbone).*$", "lr": 0.00005}]),
    lr=0.0001,
    betas=[0.9, 0.999],
    weight_decay=0.0001,
)
lr_scheduler = L(optim.lr_scheduler.MultiStepLR)(milestones=[100], gamma=0.1)
model.backbone.name = "B2"
model.backbone.use_lab = False
model.neck_depth_mult = 0.67
model.neck_out_channels = 256
model.num_classes = NUM_CLASSES
model.num_body_points = NUM_BODY_POINTS
criterion.num_classes = NUM_CLASSES
