"""Inspect a VisionHub model path and emit normalized JSON metadata."""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from visionhub.detection_variants import (
    normalize_detection_family,
    normalize_detection_variant,
    resolve_detection_config_module,
)


SUPPORTED_MODEL_SUFFIXES = {".pth", ".pt", ".ckpt", ".onnx", ".engine", ".trt"}
RUNTIME_SUFFIXES = {
    "pytorch": {".pth", ".pt", ".ckpt"},
    "onnx": {".onnx"},
    "tensorrt": {".engine", ".trt"},
}
POSE_FAMILIES = {"detrpose", "rtmopose", "rtmdetpose"}
DETECTION_FAMILIES = {"detrdet", "rtmodet", "rtmdetdet"}
DEFAULT_POSE_CONFIGS = {
    "detrpose": "configs.detrpose.detrpose_hgnetv2_s_custom",
    "rtmopose": "configs.rtmopose.rtmopose_hgnetv2_s_custom",
    "rtmdetpose": "configs.rtmdetpose.rtmdetpose_hgnetv2_s_custom",
}
PREFERRED_MODEL_NAMES = {
    "checkpoint_best_regular.pth": 0,
    "checkpoint_best.pth": 1,
    "checkpoint.pth": 2,
    "latest.pth": 3,
    "least_confused.pt": 4,
    "least_confused.pth": 5,
    "last.pt": 6,
    "last.pth": 7,
    "best.pt": 8,
    "best.pth": 9,
}


def _stderr(message: str) -> None:
    print(message, file=sys.stderr)


def _checkpoint_helpers():
    try:
        from .checkpoint_metadata import (
            load_checkpoint,
            resolve_checkpoint_architecture_metadata,
        )
    except ImportError:
        from checkpoint_metadata import (
            load_checkpoint,
            resolve_checkpoint_architecture_metadata,
        )
    return load_checkpoint, resolve_checkpoint_architecture_metadata


def _write_json(path: Path, payload: Dict[str, Any], pretty: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2 if pretty else None)
        handle.write("\n")


def _normalize_runtime(value: Optional[str]) -> str:
    runtime = str(value or "auto").strip().lower()
    aliases = {
        "torch": "pytorch",
        "pt": "pytorch",
        "onnxruntime": "onnx",
        "trt": "tensorrt",
        "engine": "tensorrt",
    }
    runtime = aliases.get(runtime, runtime)
    if runtime not in {"auto", "pytorch", "onnx", "tensorrt"}:
        raise ValueError(f"Unsupported runtime '{value}'. Use auto, pytorch, onnx, or tensorrt.")
    return runtime


def _detect_runtime_from_path(model_path: Path, runtime_hint: str) -> str:
    runtime = _normalize_runtime(runtime_hint)
    if runtime != "auto":
        return runtime

    suffix = model_path.suffix.lower()
    for detected_runtime, suffixes in RUNTIME_SUFFIXES.items():
        if suffix in suffixes:
            return detected_runtime
    raise ValueError(f"Could not infer runtime from model path: {model_path}")


def _module_exists(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _normalize_config_module(config_value: Any) -> Optional[str]:
    if config_value is None:
        return None

    value = str(config_value).strip()
    if not value:
        return None

    normalized = value.replace("\\", "/")
    if normalized.endswith(".py"):
        normalized = normalized[:-3]
    normalized = normalized.lstrip("./")

    configs_match = re.search(r"(^|/)(configs(?:/|$).*)", normalized)
    if configs_match:
        normalized = configs_match.group(2)

    if "/" in normalized:
        normalized = normalized.replace("/", ".")

    normalized = normalized.strip(".")
    if not normalized.startswith("configs."):
        return None
    return normalized


def _infer_family_from_values(values: Sequence[Any]) -> str:
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


def _candidate_metadata_json_paths(model_path: Path) -> List[Path]:
    model_dir = model_path.parent
    model_name = model_path.stem
    return [
        model_dir / f"{model_name}_class_mappings.json",
        model_dir / "class_mappings.json",
        model_dir / f"{model_name}.json",
    ]


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _load_sidecar_payload(model_path: Path) -> Tuple[Optional[Path], Dict[str, Any]]:
    for json_path in _candidate_metadata_json_paths(model_path):
        if json_path.is_file():
            return json_path, _load_json(json_path)
    return None, {}


def _path_sort_key(path: Path) -> Tuple[int, int, str]:
    try:
        depth = len(path.relative_to(path.anchor).parts)
    except ValueError:
        depth = len(path.parts)
    return (
        PREFERRED_MODEL_NAMES.get(path.name.lower(), 100),
        depth,
        str(path),
    )


def _resolve_model_path(model_value: str, runtime_hint: str) -> Path:
    candidate = Path(model_value).expanduser()
    if candidate.is_file():
        if candidate.suffix.lower() not in SUPPORTED_MODEL_SUFFIXES:
            raise FileNotFoundError(f"Unsupported model file type: {candidate}")
        return candidate.resolve()

    if not candidate.is_dir():
        raise FileNotFoundError(f"Model path not found: {candidate}")

    normalized_runtime = _normalize_runtime(runtime_hint)
    suffixes = (
        SUPPORTED_MODEL_SUFFIXES
        if normalized_runtime == "auto"
        else RUNTIME_SUFFIXES[normalized_runtime]
    )
    matches = sorted(
        (
            path.resolve()
            for path in candidate.rglob("*")
            if path.is_file() and path.suffix.lower() in suffixes
        ),
        key=_path_sort_key,
    )
    if not matches:
        raise FileNotFoundError(
            f"No supported model file found under {candidate}. "
            "Expected .pth, .pt, .ckpt, .onnx, .engine, or .trt."
        )
    return matches[0]


def _class_names_from_mapping(mapping: Mapping[Any, Any]) -> List[str]:
    ordered: List[Tuple[int, str]] = []
    for key, value in mapping.items():
        try:
            ordered.append((int(key), str(value)))
        except (TypeError, ValueError):
            continue
    return [value for _, value in sorted(ordered)]


def _normalize_skeleton_connections(raw_connections: Any) -> Dict[str, List[List[int]]]:
    if not isinstance(raw_connections, Mapping):
        return {}

    normalized: Dict[str, List[List[int]]] = {}
    for key, value in raw_connections.items():
        try:
            normalized_key = str(int(key))
        except (TypeError, ValueError):
            normalized_key = str(key)
        pairs: List[List[int]] = []
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            for item in value:
                if not isinstance(item, Sequence) or isinstance(item, (str, bytes)) or len(item) < 2:
                    continue
                try:
                    pairs.append([int(item[0]), int(item[1])])
                except (TypeError, ValueError):
                    continue
        normalized[normalized_key] = pairs
    return normalized


def _infer_task(
    *,
    family: Optional[str],
    skeleton_connections: Mapping[str, Any],
    detection_only: bool,
    classifier_hint: bool = False,
) -> str:
    if classifier_hint:
        return "classify"

    normalized_family = str(family or "").strip().lower()
    if normalized_family in DETECTION_FAMILIES:
        return "detect"
    if normalized_family in POSE_FAMILIES:
        return "pose"
    if detection_only:
        return "detect"
    if skeleton_connections:
        return "pose"
    raise ValueError("Could not infer task from model metadata.")


def _infer_variant_from_values(values: Sequence[Any]) -> Optional[str]:
    patterns = [
        (r"(?:^|[_\-.])(nano)(?:[_\-.]|$)", "n"),
        (r"(?:^|[_\-.])(small)(?:[_\-.]|$)", "s"),
        (r"(?:^|[_\-.])(medium)(?:[_\-.]|$)", "m"),
        (r"(?:^|[_\-.])(large)(?:[_\-.]|$)", "l"),
        (r"(?:^|[_\-.])(xlarge|xl|extra-large|extra_large)(?:[_\-.]|$)", "x"),
        (r"(?:^|[_\-.])hgnetv2_([nsmxl])(?:[_\-.]|$)", None),
        (r"(?:^|[_\-.])([nsmxl])(?:[_\-.]|custom|ckpt|pth|pt|onnx|engine|trt|$)", None),
    ]
    for raw_value in (str(value or "").lower() for value in values):
        for pattern, mapped in patterns:
            match = re.search(pattern, raw_value)
            if not match:
                continue
            candidate = mapped or match.group(1)
            if candidate in {"n", "s", "m", "l", "x"}:
                return candidate
    return None


def _normalize_optional_variant(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if not normalized:
        return None
    return normalize_detection_variant(normalized)


def _normalize_optional_detection_family(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if not normalized or normalized == "auto":
        return None
    return normalize_detection_family(normalized)


def _normalize_optional_pose_family(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if not normalized or normalized == "auto":
        return None
    aliases = {
        "detr": "detrpose",
        "rtmo": "rtmopose",
        "rtmdet": "rtmdetpose",
        "rtmdet-pose": "rtmdetpose",
        "rtmdet_pose": "rtmdetpose",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in POSE_FAMILIES:
        raise ValueError(
            f"Unsupported pose family '{value}'. Use one of: {', '.join(sorted(POSE_FAMILIES))}."
        )
    return normalized


def _auto_detect_detection_config_module(model_path: Path, family: str, variant: Optional[str]) -> Optional[str]:
    normalized_family = _normalize_optional_detection_family(family)
    if normalized_family is None:
        return None

    candidate_names: List[str] = []
    for candidate_dir in [model_path.parent, *model_path.parents]:
        name = candidate_dir.name
        if name and name not in candidate_names:
            candidate_names.append(name)

    family_dir = {
        "detrdet": "detrdet",
        "rtmodet": "rtmodet",
        "rtmdetdet": "rtmdetdet",
    }[normalized_family]

    for name in candidate_names:
        normalized = _normalize_config_module(name)
        if normalized and normalized.startswith(f"configs.{family_dir}.") and _module_exists(normalized):
            return normalized

    normalized_variant = normalize_detection_variant(variant or "s")
    fallback = resolve_detection_config_module(normalized_family, normalized_variant)
    if fallback and _module_exists(fallback):
        return fallback
    return None


def _auto_detect_pose_config_module(model_path: Path, family: str) -> Optional[str]:
    normalized_family = _normalize_optional_pose_family(family)
    if normalized_family is None:
        return None

    families = [normalized_family]
    config_dirs = {
        "detrpose": "detrpose",
        "rtmopose": "rtmopose",
        "rtmdetpose": "rtmdetpose",
    }

    def candidate_names(family_name: str, name: str) -> List[str]:
        names: List[str] = []
        if family_name in {"detrpose", "rtmopose"} and name.startswith(family_name):
            names.append(name)
        elif family_name == "rtmdetpose":
            if name.startswith("rtmdetpose_hgnetv2_"):
                names.append(name)
            if name.startswith("rtmdet_hgnetv2_"):
                names.append(name.replace("rtmdet_hgnetv2_", "rtmdetpose_hgnetv2_", 1))
        return names

    names_to_try: List[str] = []
    for candidate_path in [model_path.parent, *model_path.parents]:
        name = candidate_path.name
        if name and name not in names_to_try:
            names_to_try.append(name)

    for family_name in families:
        config_dir = config_dirs[family_name]
        for name in names_to_try:
            for candidate_name in candidate_names(family_name, name):
                module_name = f"configs.{config_dir}.{candidate_name}"
                if _module_exists(module_name):
                    return module_name

    default_module = DEFAULT_POSE_CONFIGS.get(normalized_family)
    if default_module and _module_exists(default_module):
        return default_module
    return None


def _extract_checkpoint_info(
    model_path: Path,
    *,
    runtime: str,
    config_hint: Optional[str],
    family_hint: Optional[str],
    no_ema: bool,
) -> Dict[str, Any]:
    load_checkpoint, resolve_checkpoint_architecture_metadata = _checkpoint_helpers()
    with contextlib.redirect_stdout(sys.stderr):
        checkpoint = load_checkpoint(str(model_path))

    is_classifier_checkpoint = (
        isinstance(checkpoint, Mapping)
        and "state_dict" in checkpoint
        and checkpoint.get("architecture") is not None
        and isinstance(checkpoint.get("class_names"), list)
        and bool(checkpoint.get("class_names"))
    )
    if is_classifier_checkpoint:
        return {
            "success": True,
            "task": "classify",
            "runtime": runtime,
            "family": "native_pytorch",
            "variant": None,
            "config_module": None,
            "checkpoint_path": str(model_path),
            "input_size": int(checkpoint.get("image_size") or 224),
            "class_names": [str(name) for name in checkpoint.get("class_names", [])],
            "skeleton_connections": {},
            "supports": {
                "image": True,
                "video": False,
            },
        }

    with contextlib.redirect_stdout(sys.stderr):
        metadata = resolve_checkpoint_architecture_metadata(
            checkpoint,
            family_hint="auto",
            use_ema=not no_ema,
        )

    skeleton_connections = _normalize_skeleton_connections(metadata.get("skeleton_connections", {}))
    detected_family = str(metadata.get("family") or "").strip().lower() or None
    if family_hint:
        if detected_family in DETECTION_FAMILIES or (
            detected_family is None and str(family_hint).strip().lower() in DETECTION_FAMILIES
        ):
            detected_family = _normalize_optional_detection_family(family_hint)
        else:
            detected_family = _normalize_optional_pose_family(family_hint)

    task = _infer_task(
        family=detected_family,
        skeleton_connections=skeleton_connections,
        detection_only=bool(metadata.get("detection_only")),
    )
    if task == "detect":
        detected_family = _normalize_optional_detection_family(detected_family)
    else:
        detected_family = _normalize_optional_pose_family(detected_family)

    variant = _normalize_optional_variant(metadata.get("variant"))
    if variant is None:
        variant = _infer_variant_from_values(
            [
                metadata.get("model_size"),
                metadata.get("config_module"),
                metadata.get("config_file"),
                model_path.name,
                model_path.parent.name,
            ]
        )

    config_module = _normalize_config_module(config_hint)
    if config_module and not _module_exists(config_module):
        raise ValueError(f"Config module not found: {config_module}")
    if config_module is None:
        checkpoint_config = _normalize_config_module(
            metadata.get("config_module") or metadata.get("config_file")
        )
        if checkpoint_config and _module_exists(checkpoint_config):
            config_module = checkpoint_config
    if config_module is None and task == "detect" and detected_family:
        config_module = _auto_detect_detection_config_module(model_path, detected_family, variant)
    if config_module is None and task == "pose" and detected_family:
        config_module = _auto_detect_pose_config_module(model_path, detected_family)

    return {
        "success": True,
        "task": task,
        "runtime": runtime,
        "family": detected_family,
        "variant": variant,
        "config_module": config_module,
        "checkpoint_path": str(model_path),
        "input_size": metadata.get("image_size"),
        "class_names": _class_names_from_mapping(metadata.get("class_mappings", {})),
        "skeleton_connections": skeleton_connections,
        "supports": {
            "image": True,
            "video": False,
        },
    }


def _extract_exported_model_info(
    model_path: Path,
    *,
    runtime: str,
    config_hint: Optional[str],
    family_hint: Optional[str],
) -> Dict[str, Any]:
    _, payload = _load_sidecar_payload(model_path)
    if not payload:
        inferred_family = str(_infer_family_from_values([str(model_path)]) or "").strip().lower() or None
        if inferred_family in DETECTION_FAMILIES | POSE_FAMILIES:
            task = "detect" if inferred_family in DETECTION_FAMILIES else "pose"
            return {
                "success": True,
                "task": task,
                "runtime": runtime,
                "family": inferred_family,
                "variant": _infer_variant_from_values([str(model_path)]),
                "config_module": _normalize_config_module(config_hint),
                "checkpoint_path": str(model_path),
                "input_size": None,
                "class_names": [],
                "skeleton_connections": {},
                "supports": {
                    "image": True,
                    "video": False,
                },
            }
        raise ValueError(
            "Could not infer task from exported model. Add the sidecar JSON produced by VisionHub export."
        )

    classifier_hint = (
        str(payload.get("model_type") or "").strip().lower() == "classification"
        or str(payload.get("model_family") or "").strip().lower() == "native_pytorch"
        or (
            payload.get("architecture") is not None
            and isinstance(payload.get("class_names"), list)
            and bool(payload.get("class_names"))
        )
    )
    if classifier_hint:
        class_mappings = payload.get("class_mappings") if isinstance(payload.get("class_mappings"), Mapping) else {}
        class_names = _class_names_from_mapping(class_mappings)
        if not class_names:
            class_names = [str(name) for name in payload.get("class_names", [])]
        return {
            "success": True,
            "task": "classify",
            "runtime": runtime,
            "family": "native_pytorch",
            "variant": None,
            "config_module": None,
            "checkpoint_path": str(model_path),
            "input_size": payload.get("image_size"),
            "class_names": class_names,
            "skeleton_connections": {},
            "supports": {
                "image": True,
                "video": False,
            },
        }

    raw_mappings = payload.get("class_mappings", {}) if isinstance(payload.get("class_mappings"), Mapping) else {}
    raw_skeletons = (
        payload.get("skeleton_connections", {}) if isinstance(payload.get("skeleton_connections"), Mapping) else {}
    )
    model_metadata = payload.get("model_metadata", {}) if isinstance(payload.get("model_metadata"), Mapping) else {}
    skeleton_connections = _normalize_skeleton_connections(raw_skeletons)

    detected_family = str(model_metadata.get("family") or "").strip().lower() or None
    if detected_family is None:
        inferred_family = str(
            _infer_family_from_values(
                [
                    str(model_path),
                    model_metadata.get("config_module"),
                    model_metadata.get("config_file"),
                    model_path.parent.name,
                ]
            )
        ).strip().lower()
        detected_family = inferred_family or None

    if family_hint:
        normalized_hint = str(family_hint).strip().lower()
        if normalized_hint in DETECTION_FAMILIES:
            detected_family = _normalize_optional_detection_family(family_hint)
        else:
            detected_family = _normalize_optional_pose_family(family_hint)

    task = _infer_task(
        family=detected_family,
        skeleton_connections=skeleton_connections,
        detection_only=bool(model_metadata.get("detection_only")),
    )
    if task == "detect":
        detected_family = _normalize_optional_detection_family(detected_family)
    else:
        detected_family = _normalize_optional_pose_family(detected_family)

    config_module = _normalize_config_module(config_hint)
    if config_module is None:
        config_module = _normalize_config_module(
            model_metadata.get("config_module") or model_metadata.get("config_file")
        )

    variant = _normalize_optional_variant(model_metadata.get("variant"))
    if variant is None:
        variant = _infer_variant_from_values(
            [
                model_metadata.get("model_size"),
                model_metadata.get("config_module"),
                model_metadata.get("config_file"),
                model_path.name,
                model_path.parent.name,
            ]
        )

    return {
        "success": True,
        "task": task,
        "runtime": runtime,
        "family": detected_family,
        "variant": variant,
        "config_module": config_module,
        "checkpoint_path": str(model_path),
        "input_size": model_metadata.get("image_size"),
        "class_names": _class_names_from_mapping(raw_mappings),
        "skeleton_connections": skeleton_connections,
        "supports": {
            "image": True,
            "video": False,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect a VisionHub model and emit normalized JSON metadata.")
    parser.add_argument("--model", "--checkpoint", "-m", dest="model", required=True)
    parser.add_argument("--runtime", default="auto")
    parser.add_argument("--family", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--json-output", dest="json_output", type=str, default=None)
    parser.add_argument("--pretty", action="store_true")
    parser.add_argument("--no-ema", action="store_true", default=False)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        model_path = _resolve_model_path(args.model, args.runtime)
        runtime = _detect_runtime_from_path(model_path, args.runtime)
        if runtime == "pytorch":
            payload = _extract_checkpoint_info(
                model_path,
                runtime=runtime,
                config_hint=args.config,
                family_hint=args.family,
                no_ema=bool(args.no_ema),
            )
        else:
            payload = _extract_exported_model_info(
                model_path,
                runtime=runtime,
                config_hint=args.config,
                family_hint=args.family,
            )

        if args.json_output:
            _write_json(Path(args.json_output).expanduser(), payload, pretty=args.pretty)

        json.dump(payload, sys.stdout, indent=2 if args.pretty else None)
        sys.stdout.write("\n")
        return 0
    except Exception as exc:
        _stderr(f"visionhub-model-info failed: {exc}")
        error_payload = {
            "success": False,
            "error": str(exc),
        }
        if getattr(args, "json_output", None):
            _write_json(Path(args.json_output).expanduser(), error_payload, pretty=getattr(args, "pretty", False))
        json.dump(error_payload, sys.stdout, indent=2 if getattr(args, "pretty", False) else None)
        sys.stdout.write("\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
