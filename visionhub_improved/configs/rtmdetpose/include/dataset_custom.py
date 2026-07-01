import os
import json

from visionhub.core import LazyCall as L
from visionhub.data import CocoDetection
from visionhub.data.dataloader import (
    BatchImageCollateFunction,
    DataLoader,
)
from visionhub.data.coco_eval import CocoEvaluator
from visionhub.data.container import Compose
import visionhub.data.transforms as T

from .rtmdetpose_hgnetv2 import eval_spatial_size

scales = [(640, 640)]
max_size = 640

train_enable_mosaic = False
train_enable_random_zoom_out = False
train_enable_random_crop = False
train_enable_random_horizontal_flip = False
train_enable_color_jitter = False
mosaic_output_size = eval_spatial_size[0] // 2

__all__ = [
    "dataset_train", "dataset_val", "dataset_test",
    "evaluator", "NUM_CLASSES", "NUM_BODY_POINTS",
    "CLASS_MAPPINGS", "CLASS_SKELETONS", "SIGMAS", "FLIP_PAIRS",
    "CATEGORY_ID_TO_CONTIGUOUS", "CONTIGUOUS_TO_CATEGORY_ID",
]

# ── Data root ─────────────────────────────────────────────────────────────────
_candidate_roots = [
    os.environ.get("RTMDETPOSE_DATA_ROOT"),
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
    DATA_ROOT = os.environ.get("RTMDETPOSE_DATA_ROOT") or os.environ.get(
        "RTMDET_DATA_ROOT", "/content/coco_data"
    )

TRAIN_DIR = os.path.join(DATA_ROOT, "train")
VAL_DIR   = os.path.join(DATA_ROOT, "val")


def _extract_7z(archive_path: str, extract_to: str) -> None:
    import subprocess
    result = subprocess.run(
        ["7z", "x", archive_path, f"-o{extract_to}", "-y"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to extract '{archive_path}'.\n"
            "7-Zip must be installed and '7z' must be on PATH.\n"
            f"stderr: {result.stderr}"
        )


def _ensure_dir(dir_path: str) -> None:
    if os.path.isdir(dir_path):
        return
    archive = dir_path + ".7z"
    if os.path.isfile(archive):
        print(f"[dataset_custom] Extracting '{archive}' ...")
        _extract_7z(archive, os.path.dirname(dir_path))
        if not os.path.isdir(dir_path):
            raise RuntimeError(
                f"Extraction succeeded but '{dir_path}' not found. "
                f"Check that the archive contains a folder named '{os.path.basename(dir_path)}'."
            )
    else:
        raise FileNotFoundError(
            f"Neither directory '{dir_path}' nor archive '{archive}' found."
        )


TRAIN_ANN = os.path.join(TRAIN_DIR, "coco_instances.json")
VAL_ANN   = os.path.join(VAL_DIR,   "coco_instances.json")
TRAIN_IMG = os.path.join(TRAIN_DIR, "images")
VAL_IMG   = os.path.join(VAL_DIR,   "images")


def _infer_flip_pairs(keypoint_names):
    index_by_name = {str(name).strip().lower(): idx for idx, name in enumerate(keypoint_names)}
    candidate_rules = [
        ("left", "right"),
        ("right", "left"),
        ("l_", "r_"),
        ("r_", "l_"),
        ("l-", "r-"),
        ("r-", "l-"),
        ("_l", "_r"),
        ("_r", "_l"),
        ("-l", "-r"),
        ("-r", "-l"),
        (".l", ".r"),
        (".r", ".l"),
        (" l", " r"),
        (" r", " l"),
    ]

    pairs = []
    seen = set()
    for idx, name in enumerate(keypoint_names):
        lower = str(name).strip().lower()
        partner_idx = None
        for src, dst in candidate_rules:
            if src not in lower:
                continue
            candidate = lower.replace(src, dst, 1)
            partner_idx = index_by_name.get(candidate)
            if partner_idx is not None and partner_idx != idx:
                break
        if partner_idx is None:
            continue
        pair = tuple(sorted((idx, partner_idx)))
        if pair in seen:
            continue
        seen.add(pair)
        pairs.append([pair[0], pair[1]])
    return pairs


def _build_train_transforms():
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
            flip_pairs=FLIP_PAIRS,
            enabled=train_enable_random_horizontal_flip,
        ),
        color_jitter=L(T.ColorJitter)(
            p=0.5,
            enabled=train_enable_color_jitter,
        ),
        resize=L(T.RandomResize)(sizes=scales, max_size=max_size),
        to_tensor=L(T.ToTensor)(),
        normalize=L(T.Normalize)(mean=[0, 0, 0], std=[1, 1, 1]),
    )


def _build_eval_transforms():
    return L(Compose)(
        resize=L(T.RandomResize)(sizes=[eval_spatial_size], max_size=max_size),
        to_tensor=L(T.ToTensor)(),
        normalize=L(T.Normalize)(mean=[0, 0, 0], std=[1, 1, 1]),
    )

# ── Dynamically derive dataset parameters ─────────────────────────────────────
if os.path.isfile(TRAIN_ANN):
    _ensure_dir(TRAIN_DIR)
    _ensure_dir(VAL_DIR)

    with open(TRAIN_ANN) as _f:
        _ann = json.load(_f)

    _cats = _ann["categories"]
    _CAT_IDS = sorted(c["id"] for c in _cats)
    CATEGORY_ID_TO_CONTIGUOUS = {
        cat_id: idx for idx, cat_id in enumerate(_CAT_IDS)
    }
    CONTIGUOUS_TO_CATEGORY_ID = {
        idx: cat_id for cat_id, idx in CATEGORY_ID_TO_CONTIGUOUS.items()
    }
    NUM_CLASSES = len(_CAT_IDS)
    NUM_BODY_POINTS = max(len(c["keypoints"]) for c in _cats)
    CLASS_MAPPINGS = {
        CATEGORY_ID_TO_CONTIGUOUS[c["id"]]: c["name"] for c in _cats
    }
    CLASS_SKELETONS = {
        CATEGORY_ID_TO_CONTIGUOUS[c["id"]]: c.get("skeleton", []) for c in _cats
    }
    _keypoint_names = next((c.get("keypoints", []) for c in _cats if c.get("keypoints")), [])
    FLIP_PAIRS = _infer_flip_pairs(_keypoint_names)

    _first_cat_with_sigmas = next(
        (c for c in _cats if "sigmas" in c), None
    )
    if _first_cat_with_sigmas is not None:
        SIGMAS = _first_cat_with_sigmas["sigmas"]
    else:
        SIGMAS = [0.05] * NUM_BODY_POINTS

else:
    print(f"[dataset_custom] Warning: Annotation file not found at {TRAIN_ANN}")
    print(f"[dataset_custom] Using placeholder values. Ensure checkpoint contains correct metadata.")
    NUM_CLASSES     = 24
    NUM_BODY_POINTS = 11
    CLASS_MAPPINGS  = {}
    CLASS_SKELETONS = {}
    CATEGORY_ID_TO_CONTIGUOUS = {}
    CONTIGUOUS_TO_CATEGORY_ID = {}
    SIGMAS = [0.05] * NUM_BODY_POINTS
    FLIP_PAIRS = []

# ── DataLoaders ───────────────────────────────────────────────────────────────

dataset_train = L(DataLoader)(
    dataset=L(CocoDetection)(
        img_folder=TRAIN_IMG,
        ann_file=TRAIN_ANN,
        category_id_to_contiguous=CATEGORY_ID_TO_CONTIGUOUS,
        num_keypoints=NUM_BODY_POINTS,
        allow_empty=True,
        transforms=_build_train_transforms(),
    ),
    total_batch_size=8,
    collate_fn=L(BatchImageCollateFunction)(
        base_size=eval_spatial_size[0],
    ),
    num_workers=2,
    shuffle=True,
    drop_last=True,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2,
)

dataset_val = L(DataLoader)(
    dataset=L(CocoDetection)(
        img_folder=VAL_IMG,
        ann_file=VAL_ANN,
        category_id_to_contiguous=CATEGORY_ID_TO_CONTIGUOUS,
        num_keypoints=NUM_BODY_POINTS,
        allow_empty=True,
        transforms=_build_eval_transforms(),
    ),
    total_batch_size=8,
    collate_fn=L(BatchImageCollateFunction)(
        base_size=eval_spatial_size[0],
    ),
    num_workers=2,
    shuffle=False,
    drop_last=False,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2,
)

dataset_test = L(DataLoader)(
    dataset=L(CocoDetection)(
        img_folder=VAL_IMG,
        ann_file=VAL_ANN,
        category_id_to_contiguous=CATEGORY_ID_TO_CONTIGUOUS,
        num_keypoints=NUM_BODY_POINTS,
        allow_empty=True,
        transforms=_build_eval_transforms(),
    ),
    total_batch_size=8,
    collate_fn=L(BatchImageCollateFunction)(
        base_size=eval_spatial_size[0],
    ),
    num_workers=2,
    shuffle=False,
    drop_last=False,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2,
)

evaluator = L(CocoEvaluator)(
    ann_file=VAL_ANN,
    iou_types=["bbox", "keypoints"],
    useCats=True,
    contiguous_to_category_id=CONTIGUOUS_TO_CATEGORY_ID,
    kpt_oks_sigmas=SIGMAS,
)
