from .include.detrdet_hgnetv2 import model, criterion, postprocessor, training_params
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

training_params["output_dir"] = "output/detrdet_hgnetv2_n_custom"
training_params["epochs"] = 100
training_params["use_ema"] = True
training_params["grad_accum_steps"] = 1

ema = L(ModelEMA)(decay=0.9999, warmups=2000)
optimizer = L(optim.AdamW)(
    params=L(get_optim_params)(cfg=[{"params": "^(?=.*backbone).*$", "lr": 0.0001}]),
    lr=0.0001,
    betas=[0.9, 0.999],
    weight_decay=0.0001,
)
lr_scheduler = L(optim.lr_scheduler.MultiStepLR)(milestones=[80], gamma=0.1)

model.backbone.name = "B0"
model.backbone.use_lab = True
model.backbone.return_idx = [1, 2, 3]
model.encoder.in_channels = [256, 512, 1024]
model.encoder.feat_strides = [8, 16, 32]
model.encoder.n_levels = 3
model.encoder.use_encoder_idx = [2]
model.encoder.depth_mult = 0.5
model.encoder.expansion = 0.34
model.encoder.hidden_dim = 128
model.encoder.dim_feedforward = 512
model.transformer.num_decoder_layers = 3
model.transformer.num_feature_levels = 3
model.transformer.dim_feedforward = 512
model.transformer.feat_strides = [8, 16, 32]
model.transformer.hidden_dim = 128
model.transformer.dec_n_points = 6
model.transformer.num_classes = NUM_CLASSES
criterion.num_classes = NUM_CLASSES
