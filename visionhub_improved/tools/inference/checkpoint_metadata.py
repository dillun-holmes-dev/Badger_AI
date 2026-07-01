"""Checkpoint metadata helpers for pose and detection inference."""

from __future__ import annotations

import importlib.util
import re
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import torch


def extract_state_dict(ckpt: Any, *, use_ema: bool = True) -> Dict[str, Any]:
    """Handle the checkpoint layouts used by VisionHub training."""
    if use_ema and isinstance(ckpt, dict) and "ema" in ckpt and ckpt["ema"] is not None:
        ema = ckpt["ema"]
        if isinstance(ema, dict):
            if "module" in ema:
                print("Using EMA weights: ckpt['ema']['module']")
                return ema["module"]
            if "state_dict" in ema:
                print("Using EMA weights: ckpt['ema']['state_dict']")
                return ema["state_dict"]
        print("Using EMA weights: ckpt['ema']")
        return ema

    if isinstance(ckpt, dict) and "model" in ckpt:
        print("Using model weights: ckpt['model']")
        return ckpt["model"]

    print("Using raw checkpoint as state_dict")
    return ckpt


def sanitize_state_dict_keys(state_dict: Mapping[str, Any]) -> Dict[str, Any]:
    """Strip common wrapper prefixes added by DDP and torch.compile."""
    cleaned: Dict[str, Any] = {}
    for key, value in state_dict.items():
        normalized = key.replace("module.", "", 1)
        if normalized.startswith("_orig_mod."):
            normalized = normalized.replace("_orig_mod.", "", 1)
        cleaned[normalized] = value
    return cleaned


def normalize_id_mapping(raw_mapping: Any) -> Dict[int, Any]:
    if not isinstance(raw_mapping, Mapping):
        return {}

    normalized: Dict[int, Any] = {}
    for key, value in raw_mapping.items():
        try:
            normalized[int(key)] = value
        except (TypeError, ValueError):
            continue
    return normalized


def load_pose_checkpoint_metadata(ckpt: Any) -> Dict[str, Any]:
    class_mappings = normalize_id_mapping(ckpt.get("class_mappings", {}) if isinstance(ckpt, dict) else {})
    skeleton_connections = normalize_id_mapping(
        ckpt.get("skeleton_connections", {}) if isinstance(ckpt, dict) else {}
    )
    contiguous_to_category_id = normalize_id_mapping(
        ckpt.get("contiguous_to_category_id", {}) if isinstance(ckpt, dict) else {}
    )
    return {
        "class_mappings": class_mappings,
        "skeleton_connections": skeleton_connections,
        "contiguous_to_category_id": contiguous_to_category_id,
    }


def load_checkpoint(checkpoint_path: str) -> Any:
    return torch.load(checkpoint_path, map_location="cpu", weights_only=False)


def normalize_config_module(config_value: Any) -> Optional[str]:
    if config_value is None:
        return None

    value = str(config_value).strip()
    if not value:
        return None

    normalized = value.replace("\\", "/")
    if normalized.endswith(".py"):
        normalized = normalized[:-3]
    normalized = normalized.lstrip("./")

    # Checkpoints may embed absolute Colab/site-packages paths such as
    # /content/venv/lib/python3.10/site-packages/configs/rtmdetdet/...
    # Only the importable module suffix starting at `configs.` is portable.
    configs_match = re.search(r"(^|/)(configs(?:/|$).*)", normalized)
    if configs_match:
        normalized = configs_match.group(2)

    if "/" in normalized:
        normalized = normalized.replace("/", ".")

    normalized = normalized.strip(".")
    if not normalized.startswith("configs."):
        return None
    return normalized


def config_module_to_file_path(config_module: Optional[str]) -> Optional[str]:
    normalized = normalize_config_module(config_module)
    if not normalized:
        return None
    return normalized.replace(".", "/") + ".py"


def infer_family_from_values(values: Sequence[Any]) -> str:
    joined = " ".join(str(value or "") for value in values).lower()
    if "detrdet" in joined or "detr_detect" in joined or "detr-detect" in joined:
        return "detrdet"
    if "rtmodet" in joined or "rtmo_detect" in joined or "rtmo-detect" in joined:
        return "rtmodet"
    if "rtmdetdet" in joined or "rtmdet_detect" in joined or "rtmdet-detect" in joined:
        return "rtmdetdet"
    if "rtmdetpose" in joined or "rtmdet_pose" in joined or "rtmdet-pose" in joined:
        return "rtmdetpose"
    if "rtmopose" in joined or "rtmo" in joined:
        return "rtmopose"
    if "rtmdet" in joined:
        return "rtmdetpose"
    if "detrpose" in joined or "detr" in joined:
        return "detrpose"
    return "auto"


def _normalize_family_name(
    family: str,
    *,
    config_module: Optional[str],
    detection_only: bool,
) -> str:
    normalized = str(family or "").strip().lower()
    if normalized in {"rtmdetdet", "rtmdet_detect", "rtmdet-detect"}:
        return "rtmdetdet"
    if normalized in {"rtmdetpose", "rtmdet_pose", "rtmdet-pose"}:
        if detection_only and config_module and config_module.startswith("configs.rtmdet."):
            return "rtmdetdet"
        return "rtmdetpose"
    if normalized == "rtmdet":
        if detection_only:
            return "rtmdetdet"
        if config_module and config_module.startswith("configs.rtmdetdet."):
            return "rtmdetdet"
        if config_module and config_module.startswith("configs.rtmdet."):
            return "rtmdetpose"
        if config_module and config_module.startswith("configs.rtmdetpose."):
            return "rtmdetpose"
    return normalized


def _canonicalize_config_module(config_module: Optional[str], family: str) -> Optional[str]:
    normalized = normalize_config_module(config_module)
    if not normalized:
        return None
    if family == "rtmdetpose" and normalized.startswith("configs.rtmdet."):
        return normalized.replace("configs.rtmdet.", "configs.rtmdetpose.", 1).replace(
            "rtmdet_hgnetv2_", "rtmdetpose_hgnetv2_", 1
        )
    if family == "rtmdetdet" and normalized.startswith("configs.rtmdet."):
        return normalized.replace("configs.rtmdet.", "configs.rtmdetdet.", 1).replace(
            "rtmdet_hgnetv2_", "rtmdetdet_hgnetv2_", 1
        )
    return normalized


def _canonicalize_config_file(config_file: Optional[str], family: str) -> Optional[str]:
    normalized_module = _canonicalize_config_module(config_file, family)
    if not normalized_module:
        return None
    return config_module_to_file_path(normalized_module)


def resolve_checkpoint_architecture_metadata(
    ckpt: Any,
    *,
    family_hint: str = "auto",
    use_ema: bool = True,
) -> Dict[str, Any]:
    model_metadata = dict(ckpt.get("model_metadata", {}) if isinstance(ckpt, dict) else {})
    legacy_args = ckpt.get("args") if isinstance(ckpt, dict) else None
    detection_only = bool(model_metadata.get("detection_only"))

    config_candidates = [
        model_metadata.get("config_module"),
        model_metadata.get("config_file"),
        getattr(legacy_args, "config_file", None),
    ]
    config_module = next(
        (normalized for normalized in (normalize_config_module(value) for value in config_candidates) if normalized),
        None,
    )

    family = str(model_metadata.get("family") or "").strip().lower() or "auto"
    family = _normalize_family_name(
        family,
        config_module=config_module,
        detection_only=detection_only,
    )
    if family == "auto":
        family = infer_family_from_values([config_module, *config_candidates, model_metadata.get("model_size")])
    family = _normalize_family_name(
        family,
        config_module=config_module,
        detection_only=detection_only,
    )
    if family_hint != "auto" and family.startswith("rtmdet"):
        family = _normalize_family_name(
            family_hint,
            config_module=None,
            detection_only=detection_only,
        )
    elif family == "auto" and family_hint != "auto":
        family = family_hint

    metadata = load_pose_checkpoint_metadata(ckpt)
    state_dict = sanitize_state_dict_keys(extract_state_dict(ckpt, use_ema=use_ema))
    if family == "auto":
        family = infer_family_from_state_dict(state_dict) or family_hint
    family = _normalize_family_name(
        family,
        config_module=config_module,
        detection_only=detection_only,
    )
    inferred_shape = infer_pose_model_shape(
        family if family != "auto" else family_hint,
        state_dict=state_dict,
        class_mappings=metadata["class_mappings"],
        skeleton_connections=metadata["skeleton_connections"],
    )

    resolved = dict(model_metadata)
    resolved.update(metadata)
    resolved["family"] = family
    resolved["config_module"] = _canonicalize_config_module(config_module, family)
    resolved["config_file"] = _canonicalize_config_file(
        model_metadata.get("config_file") or getattr(legacy_args, "config_file", None),
        family,
    )
    if resolved.get("num_classes") is None:
        resolved["num_classes"] = inferred_shape.get("num_classes")
    if resolved.get("num_body_points") is None:
        resolved["num_body_points"] = inferred_shape.get("num_body_points")
    resolved["state_dict"] = state_dict
    return resolved


def config_module_exists(config_module: Optional[str]) -> bool:
    if not config_module:
        return False
    try:
        return importlib.util.find_spec(config_module) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def disable_pretrained_backbone_downloads(cfg: Any) -> None:
    """Disable backbone-side pretrained downloads for inference-time config imports."""
    model = getattr(cfg, "model", None)
    backbone = getattr(model, "backbone", None)
    if backbone is None:
        return

    if hasattr(backbone, "pretrained"):
        backbone.pretrained = False
    if hasattr(backbone, "checkpoint"):
        backbone.checkpoint = None


def infer_family_from_state_dict(state_dict: Mapping[str, Any]) -> str:
    keys = state_dict.keys()
    if any(key.startswith("transformer.") for key in keys):
        return "detrpose"
    if any(key.startswith("dcc.") for key in keys):
        return "rtmopose"
    if any("head.kpt_vis_preds." in key for key in keys):
        return "rtmopose"
    if any("head.bbox_preds." in key for key in keys):
        return "rtmodet"
    if any("head.kpt_preds." in key for key in keys):
        return "rtmdetpose"
    if any("head.ltrb_preds." in key for key in keys):
        return "rtmdetdet"
    return "auto"


def infer_pose_model_shape(
    family: str,
    *,
    state_dict: Mapping[str, Any],
    class_mappings: Optional[Mapping[int, Any]] = None,
    skeleton_connections: Optional[Mapping[int, Any]] = None,
) -> Dict[str, Optional[int]]:
    """Infer dataset-shaped model dimensions from checkpoint contents."""
    class_mappings = class_mappings or {}
    skeleton_connections = skeleton_connections or {}

    num_classes = _infer_num_classes(family, state_dict, class_mappings)
    num_body_points = _infer_num_body_points(family, state_dict, skeleton_connections)
    return {
        "num_classes": num_classes,
        "num_body_points": num_body_points,
    }


def _infer_num_classes(
    family: str,
    state_dict: Mapping[str, Any],
    class_mappings: Mapping[int, Any],
) -> Optional[int]:
    candidates = []

    if class_mappings:
        candidates.append(max(class_mappings) + 1)

    for key in _class_shape_keys(family):
        tensor = state_dict.get(key)
        if tensor is not None and getattr(tensor, "shape", None):
            candidates.append(int(tensor.shape[0]))

    return _first_positive(candidates)


def _infer_num_body_points(
    family: str,
    state_dict: Mapping[str, Any],
    skeleton_connections: Mapping[int, Any],
) -> Optional[int]:
    candidates = []

    for skeleton in skeleton_connections.values():
        if isinstance(skeleton, Iterable):
            max_index = -1
            for connection in skeleton:
                if not isinstance(connection, Iterable):
                    continue
                for point_idx in connection:
                    try:
                        max_index = max(max_index, int(point_idx))
                    except (TypeError, ValueError):
                        continue
            if max_index >= 0:
                candidates.append(max_index + 1)

    for key, divisor in _keypoint_shape_rules(family):
        tensor = state_dict.get(key)
        if tensor is None or not getattr(tensor, "shape", None):
            continue
        candidates.append(int(tensor.shape[0] // divisor))

    return _first_positive(candidates)


def _class_shape_keys(family: str) -> Iterable[str]:
    if family == "detrpose":
        return (
            "transformer.class_embed.0.weight",
            "transformer.enc_out_class_embed.weight",
            "transformer.label_enc.weight",
        )
    return ("head.cls_preds.0.weight",)


def _keypoint_shape_rules(family: str) -> Iterable[Tuple[str, int]]:
    if family in {"detrdet", "rtmodet", "rtmdetdet"}:
        return ()
    if family == "detrpose":
        return (
            ("transformer.keypoint_embedding.weight", 1),
            ("transformer.pose_enc.weight", 1),
            ("transformer.enc_pose_embed.layers.3.weight", 2),
        )
    if family == "rtmopose":
        return (
            ("head.kpt_vis_preds.0.weight", 1),
            ("head.kpt_reg_preds.0.weight", 2),
        )
    return (
        ("head.kpt_preds.0.weight", 3),
    )


def _first_positive(values: Iterable[int]) -> Optional[int]:
    for value in values:
        if value and value > 0:
            return int(value)
    return None
