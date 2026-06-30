"""
Utilities package.
"""
from .box_ops import xywh_to_xyxy, xyxy_to_xywh, nms, scale_boxes
from .metrics import MeanAveragePrecision, compute_ap, compute_iou
from .logging import Logger, save_checkpoint, load_checkpoint, setup_experiment_dir
from .analysis import ErrorAnalyzer, COCO_CLASSES, SUPER_CATEGORIES
from .quantize import convert_to_fp16, fuse_all_conv_bn, export_onnx, quantize_int8, deploy, AutomaticMixedPrecision
from .distill import DistillationTrainer, FeatureDistiller, distillation_loss
from .tta import TTAPredictor, weighted_box_fusion, ModelEnsemble
from .benchmark import BenchmarkRunner, quick_benchmark
