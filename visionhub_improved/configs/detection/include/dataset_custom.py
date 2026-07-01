import json
import os

from visionhub.core import LazyCall as L
from visionhub.data import CocoDetection
from visionhub.data.coco_eval import CocoEvaluator
from visionhub.data.container import Compose
from visionhub.data.dataloader import BatchImageCollateFunction, DataLoader
import visionhub.data.transforms as T

eval_spatial_size = (640, 640)
scales = [(640, 640)]
max_size = 640
train_allow_empty = True

train_enable_mosaic = False
train_enable_random_zoom_out = False
train_enable_random_crop = False
train_enable_random_horizontal_flip = False
train_enable_color_jitter = False
mosaic_output_size = eval_spatial_size[0] // 2

__all__ = [
    "dataset_train",
    "dataset_val",
    "dataset_test",
    "evaluator",
    "NUM_CLASSES",
    "NUM_BODY_POINTS",
    "CLASS_MAPPINGS",
    "CLASS_SKELETONS",
    "CATEGORY_ID_TO_CONTIGUOUS",
    "CONTIGUOUS_TO_CATEGORY_ID",
]

_candidate_roots = [
    os.environ.get("DETRDET_DATA_ROOT"),
    os.environ.get("RTMODET_DATA_ROOT"),
    os.environ.get("RTMDETDET_DATA_ROOT"),
    os.environ.get("RTMDET_DATA_ROOT"),
    os.environ.get("RTMOPOSE_DATA_ROOT"),
    os.environ.get("DETRPOSE_DATA_ROOT"),
    "/content/coco_data",
    "./data/coco",
]

DATA_ROOT = None
for root in _candidate_roots:
    if root and os.path.isdir(root):
        DATA_ROOT = root
        break

if DATA_ROOT is None:
    DATA_ROOT = (
        os.environ.get("DETRDET_DATA_ROOT")
        or os.environ.get("RTMODET_DATA_ROOT")
        or os.environ.get("RTMDETDET_DATA_ROOT")
        or os.environ.get("RTMDET_DATA_ROOT")
        or "/content/coco_data"
    )

TRAIN_DIR = os.path.join(DATA_ROOT, "train")
VAL_DIR = os.path.join(DATA_ROOT, "val")

TRAIN_ANN = os.path.join(TRAIN_DIR, "coco_instances.json")
VAL_ANN = os.path.join(VAL_DIR, "coco_instances.json")

TRAIN_IMG = os.path.join(TRAIN_DIR, "images")
VAL_IMG = os.path.join(VAL_DIR, "images")

if os.path.isfile(TRAIN_ANN):
    with open(TRAIN_ANN, encoding="utf-8") as handle:
        ann = json.load(handle)

    cats = ann["categories"]
    cat_ids = sorted(c["id"] for c in cats)

    CATEGORY_ID_TO_CONTIGUOUS = {
        cat_id: idx for idx, cat_id in enumerate(cat_ids)
    }
    CONTIGUOUS_TO_CATEGORY_ID = {
        idx: cat_id for cat_id, idx in CATEGORY_ID_TO_CONTIGUOUS.items()
    }

    NUM_CLASSES = len(cat_ids)
    NUM_BODY_POINTS = 0

    CLASS_MAPPINGS = {
        CATEGORY_ID_TO_CONTIGUOUS[c["id"]]: c["name"] for c in cats
    }
    CLASS_SKELETONS = {}
else:
    print(f"[dataset_custom] Warning: Annotation file not found at {TRAIN_ANN}")
    print("[dataset_custom] Using placeholder detection metadata.")

    NUM_CLASSES = 24
    NUM_BODY_POINTS = 0
    CLASS_MAPPINGS = {}
    CLASS_SKELETONS = {}
    CATEGORY_ID_TO_CONTIGUOUS = {}
    CONTIGUOUS_TO_CATEGORY_ID = {}


def _build_train_transforms(resize_sizes):
    return L(Compose)(
        mosaic=L(T.Mosaic)(
            output_size=mosaic_output_size,
            max_size=max_size,
            probability=1.0,
            enabled=train_enable_mosaic,
        ),
        random_zoom_out=L(T.RandomZoomOut)(
            p=0.5,
            enabled=train_enable_random_zoom_out,
        ),
        random_crop=L(T.RandomCrop)(
            p=0.5,
            enabled=train_enable_random_crop,
        ),
        random_horizontal_flip=L(T.RandomHorizontalFlip)(
            p=0.5,
            enabled=train_enable_random_horizontal_flip,
        ),
        color_jitter=L(T.ColorJitter)(
            p=0.5,
            enabled=train_enable_color_jitter,
        ),
        resize=L(T.RandomResize)(
            sizes=resize_sizes,
            max_size=max_size,
        ),
        to_tensor=L(T.ToTensor)(),
        normalize=L(T.Normalize)(
            mean=[0, 0, 0],
            std=[1, 1, 1],
        ),
    )


def _build_eval_transforms(resize_sizes):
    return L(Compose)(
        resize=L(T.RandomResize)(
            sizes=resize_sizes,
            max_size=max_size,
        ),
        to_tensor=L(T.ToTensor)(),
        normalize=L(T.Normalize)(
            mean=[0, 0, 0],
            std=[1, 1, 1],
        ),
    )


def _build_dataset(img_folder, ann_file, resize_sizes, allow_empty, training):
    return L(CocoDetection)(
        img_folder=img_folder,
        ann_file=ann_file,
        category_id_to_contiguous=CATEGORY_ID_TO_CONTIGUOUS,
        num_keypoints=0,
        require_keypoints=False,
        allow_empty=allow_empty,
        transforms=(
            _build_train_transforms(resize_sizes)
            if training
            else _build_eval_transforms(resize_sizes)
        ),
    )


dataset_train = L(DataLoader)(
    dataset=_build_dataset(
        TRAIN_IMG,
        TRAIN_ANN,
        scales,
        allow_empty=train_allow_empty,
        training=True,
    ),
    total_batch_size=8,
    collate_fn=L(BatchImageCollateFunction)(
        base_size=eval_spatial_size[0],
    ),
    num_workers=4,
    shuffle=True,
    drop_last=False,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2,
)

dataset_val = L(DataLoader)(
    dataset=_build_dataset(
        VAL_IMG,
        VAL_ANN,
        [eval_spatial_size],
        allow_empty=True,
        training=False,
    ),
    total_batch_size=8,
    collate_fn=L(BatchImageCollateFunction)(
        base_size=eval_spatial_size[0],
    ),
    num_workers=4,
    shuffle=False,
    drop_last=False,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2,
)

dataset_test = L(DataLoader)(
    dataset=_build_dataset(
        VAL_IMG,
        VAL_ANN,
        [eval_spatial_size],
        allow_empty=True,
        training=False,
    ),
    total_batch_size=8,
    collate_fn=L(BatchImageCollateFunction)(
        base_size=eval_spatial_size[0],
    ),
    num_workers=4,
    shuffle=False,
    drop_last=False,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2,
)

evaluator = L(CocoEvaluator)(
    ann_file=VAL_ANN,
    iou_types=["bbox"],
    useCats=True,
    contiguous_to_category_id=CONTIGUOUS_TO_CATEGORY_ID,
)
